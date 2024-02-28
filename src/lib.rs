use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

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

#[cfg(test)]
mod tests {
    use crate::Value;

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
    }
}
