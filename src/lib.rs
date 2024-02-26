use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Debug, Clone)]
struct Value(Rc<Value_>);

#[derive(Debug)]
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

#[derive(Debug)]
enum Op {
    None,
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    Tanh(Value),
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
