use std::cell::RefCell;
use std::rc::Rc;

use ruff_python_ast::name::Name;

use super::cyclic::TypeIdentity;
use super::recursive::RecursiveOperation;
use super::tuple::{Tuple, VariableSegment};
use super::{
    ApplyTypeMappingVisitor, BindingContext, CallableType, Parameter, SelfBinding, Type,
    TypeContext, TypeMapping,
};
use crate::{Db, FxOrderMap, ProgramEnvironment};

/// A structural mapping of a finite declaration body. Keys identify original expression
/// nodes, so two nodes that produce the same closed type remain distinct.
#[derive(Debug, Default)]
pub(super) struct TypeProjection<'db> {
    nodes: RefCell<FxOrderMap<Type<'db>, Type<'db>>>,
}

impl<'db> TypeProjection<'db> {
    pub(super) fn mapping(&self) -> TypeProjectionMapping<'_, 'db> {
        TypeProjectionMapping {
            projection: self,
            previous: self.nodes.borrow().clone(),
            current: RefCell::default(),
        }
    }

    pub(super) fn into_nodes(self) -> FxOrderMap<Type<'db>, Type<'db>> {
        self.nodes.into_inner()
    }
}

pub(super) struct TypeProjectionMapping<'a, 'db> {
    projection: &'a TypeProjection<'db>,
    previous: FxOrderMap<Type<'db>, Type<'db>>,
    current: RefCell<FxOrderMap<Type<'db>, Type<'db>>>,
}

impl<'db> TypeProjectionMapping<'_, 'db> {
    pub(super) fn record(&self, source: Type<'db>, target: Type<'db>) {
        let mut current = self.current.borrow_mut();
        let mut inherited = false;
        for (&original, &previous) in &self.previous {
            if previous == source {
                current.insert(original, target);
                inherited = true;
            }
        }
        if !inherited {
            current.insert(source, target);
        }
    }
}

impl Drop for TypeProjectionMapping<'_, '_> {
    fn drop(&mut self) {
        *self.projection.nodes.borrow_mut() = std::mem::take(self.current.get_mut());
    }
}

/// A position in a structural type expression, independent of the closed value at that position.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) enum ObservationEdge {
    Identity,
    TupleElement(usize),
    TupleSuffix(usize),
    TupleVariable,
    UnionElement(usize),
    IntersectionPositive(usize),
    IntersectionNegative(usize),
    GenericArgument(usize),
    CallableParameter {
        overload: usize,
        parameter: usize,
    },
    CallableReturn {
        overload: usize,
    },
    CallableReceiver {
        overload: usize,
        relation: usize,
        annotation: bool,
    },
    TypedDictField(Name),
    TypedDictExtraItems,
    ProtocolMemberRead {
        name: Name,
        class_access: bool,
    },
    ProtocolMemberWrite {
        name: Name,
        class_access: bool,
    },
}

impl ObservationEdge {
    fn is_contravariant(&self) -> bool {
        matches!(
            self,
            Self::CallableParameter { .. }
                | Self::IntersectionNegative(_)
                | Self::ProtocolMemberWrite { .. }
        )
    }
}

/// A finite body node, or a projection into the value substituted for that node.
/// Parameter projections describe descent into an argument, not a recursive edge of the body.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct ExpressionNode<'db> {
    pub(super) template: Type<'db>,
    path: Box<[ObservationEdge]>,
}

impl<'db> From<Type<'db>> for ExpressionNode<'db> {
    fn from(template: Type<'db>) -> Self {
        Self {
            template,
            path: Box::default(),
        }
    }
}

/// The two distinct bindings performed when a previously captured callable is used.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct CallableSelfBinding<'db> {
    pub(super) receiver: Type<'db>,
    pub(super) self_type: Type<'db>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum ObservedOperationKind<'db> {
    Mapping(RecursiveOperation<'db>),
    CallableBinding {
        callable: CallableType<'db>,
        binding: CallableSelfBinding<'db>,
        path_depth: usize,
    },
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct ObservedOperation<'db> {
    operation: ObservedOperationKind<'db>,
    contravariant: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct ObservedTypeOrigin<'db> {
    pub(super) constructor: TypeIdentity<'db>,
    pub(super) application: Type<'db>,
    pub(super) node: ExpressionNode<'db>,
    pub(super) operations: Box<[ObservedOperation<'db>]>,
}

/// An explicit operand of a recursive proof. Child expressions are selected by an edge in
/// its declaration body; their identity is never inferred from an equal resulting type.
#[derive(Clone, Debug)]
pub(super) struct ObservedType<'db> {
    pub(super) ty: Type<'db>,
    origin: Option<Rc<ObservedTypeOrigin<'db>>>,
    shape: ObservedShape<'db>,
}

#[derive(Clone, Debug)]
enum ObservedShape<'db> {
    Root,
    Expression(Rc<FxOrderMap<Type<'db>, Type<'db>>>),
    Normalized(Rc<[ObservedType<'db>]>),
    Merged(Rc<[ObservedType<'db>]>),
    /// A semantic helper has not exposed the edge producing its result. Keep that dependency
    /// unresolved rather than pretending its result is an independent proof root.
    Unresolved(Rc<ObservedType<'db>>),
}

impl<'db> ObservedType<'db> {
    pub(super) fn root(ty: Type<'db>) -> Self {
        Self {
            ty,
            origin: None,
            shape: ObservedShape::Root,
        }
    }
    pub(super) fn is_normalized_union(&self) -> bool {
        matches!(self.shape, ObservedShape::Normalized(_))
    }
    pub(super) fn unresolved(&self) -> Self {
        Self {
            ty: self.ty,
            origin: self.origin.clone(),
            shape: ObservedShape::Unresolved(Rc::new(self.clone())),
        }
    }
    pub(super) fn origin(&self) -> Option<Rc<ObservedTypeOrigin<'db>>> {
        match &self.shape {
            ObservedShape::Unresolved(_) => None,
            _ => self.origin.clone(),
        }
    }

    /// An unresolved derivative still depends on its nearest observed expression. This is
    /// evidence of a recursive dependency, not the identity of the derivative itself.
    pub(super) fn dependency_origin(&self) -> Option<Rc<ObservedTypeOrigin<'db>>> {
        match &self.shape {
            ObservedShape::Unresolved(parent) => self
                .origin
                .clone()
                .or_else(|| parent.origin())
                .or_else(|| parent.dependency_origin()),
            _ => None,
        }
    }
    #[cfg(test)]
    fn is_unresolved(&self) -> bool {
        matches!(self.shape, ObservedShape::Unresolved(_))
    }

    /// Keep an unchanged operand. A changed value requires an explicit structural edge.
    pub(super) fn unchanged_or_unresolved(&self, ty: Type<'db>) -> Self {
        if self.ty == ty {
            return self.clone();
        }
        match self.shape {
            ObservedShape::Root => Self::root(ty),
            _ => Self {
                ty,
                origin: self.origin.clone(),
                shape: ObservedShape::Unresolved(Rc::new(self.clone())),
            },
        }
    }

    pub(super) fn child_at(
        &self,
        db: &'db dyn Db,
        env: &ProgramEnvironment<'db>,
        ty: Type<'db>,
        edge: ObservationEdge,
    ) -> Self {
        if edge == ObservationEdge::Identity {
            return self.unchanged_or_unresolved(ty);
        }
        if let ObservationEdge::UnionElement(index) = edge {
            return self
                .union_children(db, env)
                .get(index)
                .filter(|child| child.ty == ty)
                .cloned()
                .unwrap_or_else(|| self.unchanged_or_unresolved(ty));
        }
        match &self.shape {
            ObservedShape::Root => Self::root(ty),
            ObservedShape::Unresolved(_) => self.unchanged_or_unresolved(ty),
            ObservedShape::Normalized(_) => self.unchanged_or_unresolved(ty),
            ObservedShape::Merged(children) => {
                let children = children
                    .iter()
                    .map(|child| child.child_at(db, env, ty, edge.clone()))
                    .collect();
                self.output_edge(ty, edge).normalized(ty, children)
            }
            ObservedShape::Expression(nodes) => {
                let Some(origin) = &self.origin else {
                    return self.unchanged_or_unresolved(ty);
                };
                if let Some(template) = observation_child(db, env, origin.node.template, &edge) {
                    let mut path = origin.node.path.to_vec();
                    path.push(edge.clone());
                    let child_origin = ObservedTypeOrigin {
                        node: ExpressionNode {
                            template,
                            path: path.into(),
                        },
                        ..(**origin).clone()
                    };
                    // The closed parent already encodes variance and invariant materialization
                    // families. Observe its value before replaying a lazy declaration expression.
                    if observation_child(db, env, self.ty, &edge) == Some(ty) {
                        return Self {
                            ty,
                            origin: Some(Rc::new(child_origin)),
                            shape: ObservedShape::Expression(Rc::clone(nodes)),
                        };
                    }
                    let projection = TypeProjection::default();
                    let Some(mapped) =
                        Self::instantiate_node(db, env, &child_origin, template, &projection)
                    else {
                        return self.unchanged_or_unresolved(ty);
                    };
                    let child = Self {
                        ty: mapped,
                        origin: Some(Rc::new(child_origin)),
                        shape: ObservedShape::Expression(Rc::new(projection.into_nodes())),
                    };
                    return child.unchanged_or_unresolved(ty);
                }
                // Substitution and normalization can expose children absent from the original
                // shape. Select the actual output edge and retain its parent expression; this
                // also distinguishes descent into a substituted parameter from a body backedge.
                if observation_child(db, env, self.ty, &edge) == Some(ty) {
                    let mut path = origin.node.path.to_vec();
                    path.push(edge);
                    return Self {
                        ty,
                        origin: Some(Rc::new(ObservedTypeOrigin {
                            node: ExpressionNode {
                                template: origin.node.template,
                                path: path.into(),
                            },
                            ..(**origin).clone()
                        })),
                        shape: ObservedShape::Expression(Rc::clone(nodes)),
                    };
                }
                self.unchanged_or_unresolved(ty)
            }
        }
    }

    /// Attach normalized children in their output order while retaining the parent expression.
    pub(super) fn normalized(&self, ty: Type<'db>, children: Vec<Self>) -> Self {
        if let [child] = children.as_slice()
            && child.ty == ty
        {
            return child.clone();
        }
        let shape = if ty.is_union() {
            ObservedShape::Normalized(children.into())
        } else {
            ObservedShape::Merged(children.into())
        };
        Self {
            ty,
            origin: self.origin.clone(),
            shape,
        }
    }

    fn instantiate_node(
        db: &'db dyn Db,
        env: &ProgramEnvironment<'db>,
        origin: &ObservedTypeOrigin<'db>,
        node: Type<'db>,
        projection: &TypeProjection<'db>,
    ) -> Option<Type<'db>> {
        let contravariant = origin
            .node
            .path
            .iter()
            .filter(|edge| edge.is_contravariant())
            .count()
            % 2
            == 1;
        let mut mapped = match origin.application {
            Type::TypeAlias(alias) => alias.observe_body_node(db, node, projection, contravariant),
            Type::Recursive(recursive) => {
                recursive.observe_body_node(db, env, node, projection, contravariant)
            }
            _ => node,
        };
        for recorded in &origin.operations {
            mapped = match &recorded.operation {
                ObservedOperationKind::Mapping(operation) => operation.with_mapping(|mapping| {
                    let mapping = if contravariant != recorded.contravariant {
                        mapping.flip()
                    } else {
                        mapping
                    };
                    let mut visitor = ApplyTypeMappingVisitor::new_for_type_construction(env)
                        .with_projection(Some(projection));
                    if let RecursiveOperation::Materialize(_, bounds) = operation {
                        visitor.materialize_typevar_bounds_and_defaults = *bounds;
                    }
                    mapped.apply_type_mapping_impl(db, &mapping, TypeContext::default(), &visitor)
                }),
                ObservedOperationKind::CallableBinding {
                    callable,
                    binding,
                    path_depth,
                } => {
                    let edge = origin.node.path.get(*path_depth);
                    let (overload, receiver) = match edge {
                        None => {
                            let Type::Callable(callable) = mapped else {
                                return None;
                            };
                            mapped = Type::Callable(callable.apply_self_with_receiver(
                                db,
                                env,
                                binding.receiver,
                                binding.self_type,
                            ));
                            continue;
                        }
                        Some(
                            ObservationEdge::CallableParameter { overload, .. }
                            | ObservationEdge::CallableReturn { overload },
                        ) => (*overload, false),
                        Some(ObservationEdge::CallableReceiver { overload, .. }) => {
                            (*overload, true)
                        }
                        _ => return None,
                    };
                    let signature = callable.signatures(db).overloads.get(overload)?;
                    // Runtime receiver substitution only acts on retained receiver obligations.
                    // Parameters and returns receive the lexical Self substitution alone.
                    if receiver {
                        let mapping = TypeMapping::BindSelf(SelfBinding::new(
                            db,
                            env,
                            binding.receiver,
                            Some(BindingContext::Synthetic(env.program(db))),
                        ));
                        let visitor = ApplyTypeMappingVisitor::new_for_type_construction(env)
                            .with_projection(Some(projection));
                        mapped = mapped.apply_type_mapping_impl(
                            db,
                            &mapping,
                            TypeContext::default(),
                            &visitor,
                        );
                    }
                    let mapping = TypeMapping::BindSelf(SelfBinding::new(
                        db,
                        env,
                        binding.self_type,
                        signature.definition().map(BindingContext::Definition),
                    ));
                    let visitor = ApplyTypeMappingVisitor::new_for_type_construction(env)
                        .with_projection(Some(projection));
                    mapped.apply_type_mapping_impl(db, &mapping, TypeContext::default(), &visitor)
                }
            };
        }
        Some(mapped)
    }

    /// Apply a structural substitution to the operand and retain the same operation for
    /// declaration children that have not been observed yet.
    pub(super) fn apply_mapping(
        &self,
        db: &'db dyn Db,
        mapping: &TypeMapping<'_, 'db>,
        visitor: &ApplyTypeMappingVisitor<'_, 'db>,
    ) -> Self {
        let projection = TypeProjection::default();
        if let ObservedShape::Expression(nodes) = &self.shape {
            projection.nodes.borrow_mut().clone_from(nodes);
        }
        let mapping_visitor = visitor
            .for_new_mapping()
            .with_normalization(super::TypeNormalization::Structural)
            .with_projection(Some(&projection));
        let ty =
            self.ty
                .apply_type_mapping_impl(db, mapping, TypeContext::default(), &mapping_visitor);
        drop(mapping_visitor);
        let Some(origin) = &self.origin else {
            return Self::root(ty);
        };
        let operation = match mapping {
            TypeMapping::Materialize(kind) => Some(RecursiveOperation::Materialize(
                *kind,
                visitor.materialize_typevar_bounds_and_defaults,
            )),
            _ => RecursiveOperation::substitution(mapping),
        };
        let Some(operation) = operation else {
            return self.unchanged_or_unresolved(ty);
        };
        let mut operations = origin.operations.to_vec();
        operations.push(ObservedOperation {
            operation: ObservedOperationKind::Mapping(operation),
            contravariant: origin
                .node
                .path
                .iter()
                .filter(|edge| edge.is_contravariant())
                .count()
                % 2
                == 1,
        });
        Self {
            ty,
            origin: Some(Rc::new(ObservedTypeOrigin {
                operations: operations.into(),
                ..(**origin).clone()
            })),
            shape: ObservedShape::Expression(Rc::new(projection.into_nodes())),
        }
    }

    /// Bind a callable with the same domain-sensitive operation used by signature checking.
    /// Keep its input signatures so later parameter, return, and receiver edges replay exactly
    /// the substitutions that applied at their original signature position.
    pub(super) fn bind_callable_self(
        &self,
        db: &'db dyn Db,
        env: &ProgramEnvironment<'db>,
        binding: CallableSelfBinding<'db>,
    ) -> Self {
        let Type::Callable(callable) = self.ty else {
            return self.unresolved();
        };
        let ty = Type::Callable(callable.apply_self_with_receiver(
            db,
            env,
            binding.receiver,
            binding.self_type,
        ));
        let Some(origin) = &self.origin else {
            return Self::root(ty);
        };
        let mut operations = origin.operations.to_vec();
        operations.push(ObservedOperation {
            operation: ObservedOperationKind::CallableBinding {
                callable,
                binding,
                path_depth: origin.node.path.len(),
            },
            contravariant: false,
        });
        Self {
            ty,
            origin: Some(Rc::new(ObservedTypeOrigin {
                operations: operations.into(),
                ..(**origin).clone()
            })),
            shape: self.shape.clone(),
        }
    }

    fn output_edge(&self, ty: Type<'db>, edge: ObservationEdge) -> Self {
        let origin = self
            .origin
            .as_ref()
            .map(|origin| {
                let mut path = origin.node.path.to_vec();
                path.push(edge);
                ObservedTypeOrigin {
                    node: ExpressionNode {
                        template: origin.node.template,
                        path: path.into(),
                    },
                    ..(**origin).clone()
                }
            })
            .map(Rc::new);
        Self {
            ty,
            origin,
            shape: self.shape.clone(),
        }
    }

    /// Build the output edges from the mapping's constituent expressions. Flattening or
    /// reordering does not change the identity of an originating node, and merged values
    /// preserve every input edge as an explicit compound observation.
    pub(super) fn union_children(
        &self,
        db: &'db dyn Db,
        env: &ProgramEnvironment<'db>,
    ) -> Vec<Self> {
        let Type::Union(union) = self.ty else {
            return Vec::new();
        };
        if let ObservedShape::Normalized(children) = &self.shape {
            return children.to_vec();
        }
        let mut candidates = Vec::new();
        if let ObservedShape::Expression(_) = &self.shape
            && let Some(origin) = &self.origin
            && let Type::Union(template) = origin.node.template
        {
            for (index, &node) in template.elements(db).iter().enumerate() {
                let mut path = origin.node.path.to_vec();
                path.push(ObservationEdge::UnionElement(index));
                let child_origin = ObservedTypeOrigin {
                    node: ExpressionNode {
                        template: node,
                        path: path.into(),
                    },
                    ..(**origin).clone()
                };
                let projection = TypeProjection::default();
                let Some(ty) = Self::instantiate_node(db, env, &child_origin, node, &projection)
                else {
                    // A pending operation could not be projected through this expression.
                    // The closed union still supplies every alternative; retain those values
                    // and their dependency rather than discarding an unobserved branch.
                    return union
                        .elements(db)
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(index, ty)| {
                            self.output_edge(ty, ObservationEdge::UnionElement(index))
                                .unresolved()
                        })
                        .collect();
                };
                let child = Self {
                    ty,
                    origin: Some(Rc::new(child_origin)),
                    shape: ObservedShape::Expression(Rc::new(projection.into_nodes())),
                };
                if child.ty.is_union() {
                    candidates.extend(child.union_children(db, env));
                } else {
                    candidates.push(child);
                }
            }
        } else {
            return union
                .elements(db)
                .iter()
                .copied()
                .enumerate()
                .map(|(index, ty)| match self.shape {
                    ObservedShape::Root => Self::root(ty),
                    _ => self.output_edge(ty, ObservationEdge::UnionElement(index)),
                })
                .collect();
        }
        union
            .elements(db)
            .iter()
            .copied()
            .enumerate()
            .map(|(index, ty)| {
                let contributors: Vec<_> = candidates
                    .iter()
                    .filter(|child| child.ty == ty)
                    .cloned()
                    .collect();
                let output = self.output_edge(ty, ObservationEdge::UnionElement(index));
                if contributors.is_empty() {
                    output
                } else {
                    output.normalized(ty, contributors)
                }
            })
            .collect()
    }

    pub(super) fn unfold(&self, db: &'db dyn Db, env: &ProgramEnvironment<'db>) -> Option<Self> {
        let projection = TypeProjection::default();
        let (template, body) = match self.ty {
            Type::TypeAlias(alias) => (
                Some(alias.raw_value_type(db)),
                alias.value_type_with_projection(db, &projection),
            ),
            Type::Recursive(recursive) => (
                recursive.observation_body(db),
                recursive
                    .unfold_with_projection(db, env, Some(&projection))
                    .into_unfolded()?,
            ),
            _ => return None,
        };
        let Some(template) = template else {
            // Opaque declaration constructors retain their closed application; their schema or
            // callable observer supplies member edges when those requirements are requested.
            return Some(Self {
                ty: body,
                origin: Some(Rc::new(ObservedTypeOrigin {
                    constructor: self.ty.to_type_identity(db),
                    application: self.ty,
                    node: match self.ty {
                        Type::Recursive(recursive) => {
                            Type::Recursive(recursive.constructor(db)).into()
                        }
                        _ => self.ty.into(),
                    },
                    operations: Box::default(),
                })),
                shape: ObservedShape::Expression(Rc::new(projection.into_nodes())),
            });
        };
        Some(Self {
            ty: body,
            origin: Some(Rc::new(ObservedTypeOrigin {
                constructor: self.ty.to_type_identity(db),
                application: self.ty,
                node: template.into(),
                operations: Box::default(),
            })),
            shape: ObservedShape::Expression(Rc::new(projection.into_nodes())),
        })
    }
}

/// The explicit expressions currently being compared. Reversing a comparison also reverses
/// these operands; a child can never borrow provenance from an equal value on the other side.
#[derive(Clone, Debug, Default)]
pub(super) struct ObservedTypePair<'db>(Option<(ObservedType<'db>, ObservedType<'db>)>);

impl<'db> ObservedTypePair<'db> {
    pub(super) fn new(source: ObservedType<'db>, target: ObservedType<'db>) -> Self {
        Self(Some((source, target)))
    }
    pub(super) fn children(
        &self,
        source: Type<'db>,
        target: Type<'db>,
    ) -> (ObservedType<'db>, ObservedType<'db>) {
        self.0.as_ref().map_or_else(
            || (ObservedType::root(source), ObservedType::root(target)),
            |(left, right)| {
                (
                    left.unchanged_or_unresolved(source),
                    right.unchanged_or_unresolved(target),
                )
            },
        )
    }
    pub(super) fn children_at(
        &self,
        db: &'db dyn Db,
        env: &ProgramEnvironment<'db>,
        source: Type<'db>,
        target: Type<'db>,
        source_edge: ObservationEdge,
        target_edge: ObservationEdge,
    ) -> (ObservedType<'db>, ObservedType<'db>) {
        self.0.as_ref().map_or_else(
            || (ObservedType::root(source), ObservedType::root(target)),
            |(left, right)| {
                (
                    left.child_at(db, env, source, source_edge),
                    right.child_at(db, env, target, target_edge),
                )
            },
        )
    }
    pub(super) fn map(
        &self,
        db: &'db dyn Db,
        source_mapping: Option<&TypeMapping<'_, 'db>>,
        target_mapping: Option<&TypeMapping<'_, 'db>>,
        visitor: &ApplyTypeMappingVisitor<'_, 'db>,
    ) -> Self {
        Self(self.0.as_ref().map(|(source, target)| {
            (
                source_mapping.map_or_else(
                    || source.clone(),
                    |mapping| source.apply_mapping(db, mapping, visitor),
                ),
                target_mapping.map_or_else(
                    || target.clone(),
                    |mapping| target.apply_mapping(db, mapping, visitor),
                ),
            )
        }))
    }
    pub(super) fn bind_callable_self(
        &self,
        db: &'db dyn Db,
        env: &ProgramEnvironment<'db>,
        source_binding: Option<CallableSelfBinding<'db>>,
        target_binding: Option<CallableSelfBinding<'db>>,
    ) -> Self {
        Self(self.0.as_ref().map(|(source, target)| {
            (
                source_binding.map_or_else(
                    || source.clone(),
                    |binding| source.bind_callable_self(db, env, binding),
                ),
                target_binding.map_or_else(
                    || target.clone(),
                    |binding| target.bind_callable_self(db, env, binding),
                ),
            )
        }))
    }

    pub(super) fn reversed(&self) -> Self {
        Self(
            self.0
                .as_ref()
                .map(|(left, right)| (right.clone(), left.clone())),
        )
    }
    pub(super) fn source_twice(&self) -> Self {
        Self(
            self.0
                .as_ref()
                .map(|(left, _)| (left.clone(), left.clone())),
        )
    }
    pub(super) fn target_twice(&self) -> Self {
        Self(
            self.0
                .as_ref()
                .map(|(_, right)| (right.clone(), right.clone())),
        )
    }
}

fn observation_child<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    template: Type<'db>,
    edge: &ObservationEdge,
) -> Option<Type<'db>> {
    match (template, edge) {
        (Type::Union(union), ObservationEdge::UnionElement(index)) => {
            union.elements(db).get(*index).copied()
        }
        (Type::Intersection(intersection), ObservationEdge::IntersectionPositive(index)) => {
            intersection.positive(db).get_index(*index).copied()
        }
        (Type::Intersection(intersection), ObservationEdge::IntersectionNegative(index)) => {
            intersection.negative(db).iter().nth(*index).copied()
        }
        (
            _,
            ObservationEdge::TupleElement(_)
            | ObservationEdge::TupleSuffix(_)
            | ObservationEdge::TupleVariable,
        ) => {
            let tuple = template.exact_tuple_instance_spec(db)?;
            match (tuple.as_ref(), edge) {
                (Tuple::Fixed(elements), ObservationEdge::TupleElement(index)) => {
                    elements.all_elements().get(*index).copied()
                }
                (Tuple::Fixed(elements), ObservationEdge::TupleSuffix(index)) => {
                    elements.all_elements().iter().rev().nth(*index).copied()
                }
                (Tuple::Variable(elements), ObservationEdge::TupleElement(index)) => {
                    elements.prefix_elements().get(*index).copied()
                }
                (Tuple::Variable(elements), ObservationEdge::TupleSuffix(index)) => {
                    elements.suffix_elements().iter().rev().nth(*index).copied()
                }
                (Tuple::Variable(elements), ObservationEdge::TupleVariable) => {
                    match elements.variable() {
                        VariableSegment::Homogeneous(ty) => Some(ty),
                        VariableSegment::TypeVarTuple(typevar) => Some(Type::TypeVar(typevar)),
                    }
                }
                _ => None,
            }
        }
        (_, ObservationEdge::GenericArgument(index)) => template
            .class_specialization(db, env)
            .and_then(|(_, specialization)| specialization.types(db).get(*index).copied()),
        (
            _,
            ObservationEdge::CallableReturn { overload }
            | ObservationEdge::CallableParameter { overload, .. }
            | ObservationEdge::CallableReceiver { overload, .. },
        ) => {
            let signatures = match template {
                Type::Callable(callable) => callable.signatures(db),
                Type::FunctionLiteral(function) => function.signature(db),
                Type::BoundMethod(method) => method.bound_signatures(db)?,
                _ => return None,
            };
            let signature = signatures.overloads.get(*overload)?;
            match edge {
                ObservationEdge::CallableReturn { .. } => Some(signature.return_ty),
                ObservationEdge::CallableParameter { parameter, .. } => signature
                    .parameters()
                    .get(*parameter)
                    .map(Parameter::annotated_type),
                ObservationEdge::CallableReceiver {
                    relation,
                    annotation,
                    ..
                } => signature
                    .receiver_relation_at(*relation)
                    .map(|(receiver, expected)| if *annotation { expected } else { receiver }),
                _ => None,
            }
        }
        (
            _,
            edge @ (ObservationEdge::ProtocolMemberRead { .. }
            | ObservationEdge::ProtocolMemberWrite { .. }),
        ) => template
            .as_protocol_instance(db)?
            .interface(db)
            .observation_child(db, edge),
        (Type::TypedDict(typed_dict), edge) => typed_dict.observation_child(db, env, edge),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use ruff_db::files::system_path_to_file;
    use ruff_db::system::DbWithWritableSystem;
    use ty_python_core::ProgramFile;

    use super::{
        CallableSelfBinding, ObservationEdge, ObservedShape, ObservedType, ObservedTypeOrigin,
        TypeProjection,
    };
    use crate::db::tests::setup_db;
    use crate::place::global_symbol;
    use crate::types::callable::CallableTypeKind;
    use crate::types::cyclic::TypeIdentity;
    use crate::types::signatures::CallableSignature;
    use crate::types::{
        BindingContext, BoundTypeVarInstance, CallableType, KnownClass, Parameter, Parameters,
        Signature, Type,
    };

    #[test]
    fn equal_outputs_retain_distinct_mapping_nodes() {
        let projection = TypeProjection::default();
        let parameter = Type::unknown();
        let literal = Type::object();
        {
            let mapping = projection.mapping();
            mapping.record(parameter, literal);
            mapping.record(literal, literal);
        }
        let nodes = projection.into_nodes();
        assert_eq!(nodes.get(&parameter), Some(&literal));
        assert_eq!(nodes.get(&literal), Some(&literal));
    }

    #[test]
    fn callable_binding_preserves_receiver_and_parameter_domains() {
        let db = setup_db();
        let env = db.program_environment();
        let self_type = Type::TypeVar(BoundTypeVarInstance::synthetic_self(
            &db,
            Type::object(),
            BindingContext::Synthetic(env.program(&db)),
        ));
        let signature = Signature::new(
            Parameters::standard([
                Parameter::positional_only(None)
                    .with_annotated_type(KnownClass::Type.to_instance(&db, &env)),
                Parameter::positional_only(None).with_annotated_type(self_type),
            ]),
            self_type,
        )
        .bind_self(&db, &env, None);
        assert_eq!(signature.receiver_relations().count(), 1);
        let callable = CallableType::new(
            &db,
            CallableSignature::single(signature),
            CallableTypeKind::Regular,
        );
        let ty = Type::Callable(callable);
        let observation = ObservedType {
            ty,
            origin: Some(Rc::new(ObservedTypeOrigin {
                constructor: TypeIdentity::Other(ty),
                application: ty,
                node: ty.into(),
                operations: Box::default(),
            })),
            shape: ObservedShape::Expression(Rc::default()),
        };
        let instance = KnownClass::Int.to_instance(&db, &env);
        let class = KnownClass::Int.to_class_literal(&db, &env);
        let bound = observation.bind_callable_self(
            &db,
            &env,
            CallableSelfBinding {
                receiver: class,
                self_type: instance,
            },
        );
        for (edge, expected) in [
            (
                ObservationEdge::CallableParameter {
                    overload: 0,
                    parameter: 0,
                },
                instance,
            ),
            (ObservationEdge::CallableReturn { overload: 0 }, instance),
            (
                ObservationEdge::CallableReceiver {
                    overload: 0,
                    relation: 0,
                    annotation: false,
                },
                class,
            ),
        ] {
            let child = bound.child_at(&db, &env, expected, edge);
            assert!(!child.is_unresolved());
            let origin = child.origin().unwrap();
            assert_eq!(
                ObservedType::instantiate_node(
                    &db,
                    &env,
                    &origin,
                    self_type,
                    &TypeProjection::default(),
                ),
                Some(expected),
            );
        }
    }

    #[test]
    fn discharging_a_receiver_keeps_later_expression_positions() {
        let db = setup_db();
        let env = db.program_environment();
        let class = KnownClass::Int.to_class_literal(&db, &env);
        let any_class = KnownClass::Type.to_instance(&db, &env);
        let signature = Signature::new(
            Parameters::standard([
                Parameter::positional_only(None).with_annotated_type(class),
                Parameter::positional_only(None).with_annotated_type(any_class),
            ]),
            Type::Never,
        )
        .bind_self(&db, &env, None)
        .bind_self(&db, &env, None);
        assert_eq!(signature.receiver_relations().count(), 2);
        let callable = CallableType::new(
            &db,
            CallableSignature::single(signature),
            CallableTypeKind::Regular,
        );
        let ty = Type::Callable(callable);
        let observation = ObservedType {
            ty,
            origin: Some(Rc::new(ObservedTypeOrigin {
                constructor: TypeIdentity::Other(ty),
                application: ty,
                node: ty.into(),
                operations: Box::default(),
            })),
            shape: ObservedShape::Expression(Rc::default()),
        };
        let bound = observation.bind_callable_self(
            &db,
            &env,
            CallableSelfBinding {
                receiver: class,
                self_type: KnownClass::Int.to_instance(&db, &env),
            },
        );
        let discharged = ObservationEdge::CallableReceiver {
            overload: 0,
            relation: 0,
            annotation: true,
        };
        let surviving = ObservationEdge::CallableReceiver {
            overload: 0,
            relation: 1,
            annotation: true,
        };
        assert_eq!(
            super::observation_child(&db, &env, bound.ty, &discharged),
            None
        );
        assert_eq!(
            super::observation_child(&db, &env, bound.ty, &surviving),
            Some(any_class)
        );
        let child = bound.child_at(&db, &env, any_class, surviving);
        assert!(!child.is_unresolved());
        let origin = child.origin().unwrap();
        assert_eq!(origin.node.template, any_class);
        assert_eq!(
            ObservedType::instantiate_node(
                &db,
                &env,
                &origin,
                origin.node.template,
                &TypeProjection::default()
            ),
            Some(any_class),
        );
    }

    #[test]
    fn tuple_edges_distinguish_a_substituted_parameter_from_an_equal_literal() {
        let mut db = setup_db();
        db.write_dedented(
            "/src/a.py",
            "type Pair[T] = tuple[T, int]\npair: Pair[int]\n",
        )
        .unwrap();
        let env = db.program_environment();
        let file = system_path_to_file(&db, "/src/a.py").unwrap();
        let file = ProgramFile::new(&db, file, env.program(&db));
        let pair = global_symbol(&db, file, "pair").place.expect_type();
        let observed = ObservedType::root(pair).unfold(&db, &env).unwrap();
        let int = KnownClass::Int.to_instance(&db, &env);
        let first = observed.child_at(&db, &env, int, ObservationEdge::TupleElement(0));
        let second = observed.child_at(&db, &env, int, ObservationEdge::TupleElement(1));
        assert_eq!(first.ty, second.ty);
        assert!(!first.is_unresolved());
        assert!(!second.is_unresolved());
        let first = first.origin().unwrap();
        let second = second.origin().unwrap();
        assert_ne!(first.node, second.node);
        assert!(first.node.template.is_type_var());
        assert_eq!(second.node.template, int);
    }

    #[test]
    fn materialized_callable_children_keep_their_opposite_variance() {
        let mut db = setup_db();
        db.write_dedented(
            "/src/a.py",
            r#"
from typing import Any, Callable
from ty_extensions import Top

type Callback = Callable[[Any], Any]
callback: Top[Callback]
"#,
        )
        .unwrap();
        let env = db.program_environment();
        let file = system_path_to_file(&db, "/src/a.py").unwrap();
        let file = ProgramFile::new(&db, file, env.program(&db));
        let callback = global_symbol(&db, file, "callback").place.expect_type();
        let observed = ObservedType::root(callback).unfold(&db, &env).unwrap();
        let parameter = observed.child_at(
            &db,
            &env,
            Type::Never,
            ObservationEdge::CallableParameter {
                overload: 0,
                parameter: 0,
            },
        );
        let result = observed.child_at(
            &db,
            &env,
            Type::object(),
            ObservationEdge::CallableReturn { overload: 0 },
        );
        assert!(!parameter.is_unresolved());
        assert!(!result.is_unresolved());
        assert_ne!(
            parameter.origin().unwrap().node,
            result.origin().unwrap().node
        );
        assert_eq!(parameter.ty, Type::Never);
        assert_eq!(result.ty, Type::object());
    }

    #[test]
    fn generic_children_keep_declared_variance_and_invariant_families() {
        let mut db = setup_db();
        db.write_dedented(
            "/src/a.py",
            r#"
from typing import Any, Generic, TypeVar
from ty_extensions import Top
T = TypeVar("T", contravariant=True)
class Consumer(Generic[T]):
    def consume(self, value: T) -> None: ...
class Cell[T]:
    value: T

type Consumed[T] = Consumer[T]
type Stored[T] = Cell[T]
consumer: Top[Consumed[Any]]
cell: Top[Stored[Any]]
"#,
        )
        .unwrap();
        let env = db.program_environment();
        let file = system_path_to_file(&db, "/src/a.py").unwrap();
        let file = ProgramFile::new(&db, file, env.program(&db));
        for (name, expected) in [("consumer", Type::Never), ("cell", Type::any())] {
            let ty = global_symbol(&db, file, name).place.expect_type();
            let observed = ObservedType::root(ty).unfold(&db, &env).unwrap();
            let child = observed.child_at(&db, &env, expected, ObservationEdge::GenericArgument(0));
            assert!(!child.is_unresolved(), "{name}");
            assert_eq!(child.ty, expected);
            if name == "cell" {
                assert!(
                    observed
                        .ty
                        .class_specialization(&db, &env)
                        .unwrap()
                        .1
                        .materialization_kind(&db)
                        .is_some()
                );
            }
        }
    }

    #[test]
    fn union_output_order_preserves_the_parameter_expression() {
        let mut db = setup_db();
        db.write_dedented(
            "/src/a.py",
            "type First[T] = T | int\ntype Last[T] = int | T\nfirst: First[str]\nlast: Last[str]\n",
        )
        .unwrap();
        let env = db.program_environment();
        let file = system_path_to_file(&db, "/src/a.py").unwrap();
        let file = ProgramFile::new(&db, file, env.program(&db));
        for name in ["first", "last"] {
            let ty = global_symbol(&db, file, name).place.expect_type();
            let observed = ObservedType::root(ty).unfold(&db, &env).unwrap();
            let str = KnownClass::Str.to_instance(&db, &env);
            let children = observed.union_children(&db, &env);
            let child = children.iter().find(|child| child.ty == str).unwrap();
            assert!(child.origin().unwrap().node.template.is_type_var());
        }
    }
}
