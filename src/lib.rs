use std::cell::RefCell;
use std::collections::{HashSet, LinkedList};
use std::fmt::Formatter;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dot_structures::*;
use graphviz_rust::dot_generator::{attr, edge, id, node, node_id};
use graphviz_rust::dot_structures::Graph;

#[derive(Clone)]
struct Value(Rc<Value_>);

impl Deref for Value {
    type Target = Value_;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Value {
    pub fn new(data: f32) -> Self {
        Value(Rc::new(Value_::new(data)))
    }

    pub fn tanh(&self) -> Value {
        let d = self.get_data().tanh();
        let mut v = Value_::new(d);
        let op = Op::Tanh((*self).clone());
        v.op = op;
        Value(Rc::new(v))
    }
}

struct Value_ {
    // unique id
    id: usize,

    // numerical representation of Value_
    data: RefCell<f32>,

    // derivative of the loss w.r.t value_
    grad: RefCell<f32>,

    // math operations that generates this Value_
    op: Op,
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Op::None => {
                write!(f, "")
            }
            Op::Add(_, _) => {
                write!(f, "{}", "+")
            }
            Op::Mul(_, _) => {
                write!(f, "{}", "x")
            }
            Op::Tanh(_) => {
                write!(f, "{}", "tanh")
            }
            Op::Sub(_, _) => {
                write!(f, "{}", "-")
            }
        }
    }
}

static ID: AtomicUsize = AtomicUsize::new(0);

fn get_id() -> usize {
    ID.fetch_add(1, Ordering::Relaxed)
}

impl Value_ {
    pub fn new(data: f32) -> Self {
        Value_ {
            data: RefCell::new(data),
            grad: RefCell::new(0.0),
            op: Op::None,
            id: get_id(),
        }
    }

    pub fn get_data(&self) -> f32 {
        *self.data.borrow()
    }

    pub fn get_grad(&self) -> f32 {
        *self.grad.borrow()
    }
}

fn topological_order(root: &Value) -> LinkedList<Value> {
    let mut order = LinkedList::new();
    let mut visited = HashSet::new();
    fn depth_first(value: &Value, visited: &mut HashSet<usize>, order: &mut LinkedList<Value>) {
        if !visited.contains(&value.id) {
            visited.insert(value.id);
            match &value.op {
                Op::None => {}
                Op::Add(v1, v2) => {
                    depth_first(v1, visited, order);
                    depth_first(v2, visited, order);
                }
                Op::Mul(v1, v2) => {
                    depth_first(v1, visited, order);
                    depth_first(v2, visited, order);
                }
                Op::Tanh(v1) => {
                    depth_first(v1, visited, order);
                }
                Op::Sub(v1, v2) => {
                    depth_first(v1, visited, order);
                    depth_first(v2, visited, order);
                }
            }
            // order has been filled with all children in topological order
            order.push_front(value.clone())
        }
    }
    depth_first(&root, &mut visited, &mut order);
    order
}

enum Op {
    None,
    Add(Value, Value),
    Mul(Value, Value),
    Tanh(Value),
    Sub(Value, Value),
}

impl std::ops::Add for &Value {
    type Output = Value;
    fn add(self, rhs: Self) -> Self::Output {
        let d = self.get_data() + rhs.get_data();
        let mut v = Value_::new(d);
        let op = Op::Add((*self).clone(), (*rhs).clone());
        v.op = op;
        Value(Rc::new(v))
    }
}

impl std::ops::Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: Self) -> Self::Output {
        let d = self.get_data() * rhs.get_data();
        let mut v = Value_::new(d);
        let op = Op::Mul((*self).clone(), (*rhs).clone());
        v.op = op;
        Value(Rc::new(v))
    }
}

impl std::ops::Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: Self) -> Self::Output {
        let d = self.get_data() - rhs.get_data();
        let mut v = Value_::new(d);
        let op = Op::Sub((*self).clone(), (*rhs).clone());
        v.op = op;
        Value(Rc::new(v))
    }
}

fn viz_computation_graph(value: &Value, graph: &mut Graph) {
    let tp_order = topological_order(value);

    for value in &tp_order {
        let value_node_id = value.id;
        let value_node = node!(
            value_node_id,
            vec![
                attr!("label", esc format!("{} | data={} grad={} op={}", value.id, value.get_data(), value.get_grad(), value.op))
            ]
        );
        graph.add_stmt(value_node.into());
        let mut add_edge = |v: &Value| {
            let p_node_id = v.id;
            let e = edge!(node_id!(p_node_id) => node_id!(value_node_id));
            graph.add_stmt(e.into());
        };
        match &value.op {
            Op::None => {}
            Op::Add(v1, v2) => {
                add_edge(v1);
                add_edge(v2);
            }
            Op::Mul(v1, v2) => {
                add_edge(v1);
                add_edge(v2);
            }
            Op::Tanh(v1) => {
                add_edge(v1);
            }
            Op::Sub(v1, v2) => {
                add_edge(v1);
                add_edge(v2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use dot_generator::*;
    use dot_structures::*;
    use graphviz_rust::cmd::CommandArg::Output;
    use graphviz_rust::cmd::Format;
    use graphviz_rust::exec;
    use graphviz_rust::printer::PrinterContext;

    use crate::{viz_computation_graph, Value};

    #[test]
    fn it_works() {
        let v1 = Value::new(0.5);
        assert_eq!(v1.get_data(), 0.5);
        let v2 = Value::new(0.3);
        let v3 = &v1 + &v2;
        assert_eq!(v3.get_data(), 0.8);
        let v4 = &v3 - &v2; // 0.8 - 0.3 = 0.5
        let v5 = &v4 * &v2; // 0.5 * 0.3 = 0.15
        assert_eq!(v5.get_data(), 0.15);
        let v6 = &v5.tanh();
        assert_eq!(v6.get_data(), 0.15f32.tanh());

        let mut graph = graph!(id!("computation"));
        viz_computation_graph(&v6, &mut graph);
        let _graph_svg = exec(
            graph,
            &mut PrinterContext::default(),
            vec![Format::Png.into(), Output("./1.png".into())],
        )
        .unwrap();
    }
}
