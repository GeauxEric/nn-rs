use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

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
}

#[derive(Debug)]
enum Op {
    None,
}

#[cfg(test)]
mod tests {
    use crate::Value;

    #[test]
    fn it_works() {
        let v1 = Value::new(0.5);
        assert_eq!(v1.get_data(), 0.5);
    }
}
