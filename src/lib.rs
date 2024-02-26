use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

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


fn get_id() -> usize {
    static ID: AtomicUsize = AtomicUsize::new(0);
    return ID.fetch_add(1, Ordering::Relaxed);
}

impl Value_ {
    pub fn new(data: f32) -> Self {
        Value_ {
            id: get_id(),
            data: RefCell::new(data),
            grad: RefCell::new(0.0),
            op: Op::None,
        }
    }

    pub fn get_data(&self) -> f32 {
        *self.data.borrow()
    }
}

impl Deref for Value {
    type Target = Value_;

    fn deref(&self) -> &Self::Target {
        return &self.0
    }
}

impl Value {
    pub fn new(data: f32) -> Self {
        let v = Value_::new(data);
        Value(Rc::new(v))
    }
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
    use crate::Value;

    #[test]
    fn it_works() {
        let v1 = Value::new(1.0);
        assert_eq!(1.0, v1.get_data());
    }
}
