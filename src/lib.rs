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
    ID.fetch_add(1, Ordering::Relaxed)
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

impl Value {
    pub fn new(data: f32) -> Self {
        Value(Rc::new(Value_::new(data)))
    }

    pub fn tanh(&self) -> Self {
        let d= self.get_data().tanh();
        let mut v = Value_::new(d);
        v.op = Op::Tanh((*self).clone());
        Value(Rc::new(v))
    }
}

impl Deref for Value {
    type Target = Value_;
    fn deref(&self) -> &Self::Target {
        &self.0
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

impl std::ops::Add for &Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        let d = self.get_data() + rhs.get_data();
        let mut v = Value_::new(d);
        v.op = Op::Add((*self).clone(), (*rhs).clone());
        Value(Rc::new(v))
    }
}

impl std::ops::Sub for &Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        let d = self.get_data() - rhs.get_data();
        let mut v = Value_::new(d);
        v.op = Op::Sub((*self).clone(), (*rhs).clone());
        Value(Rc::new(v))
    }
}

impl std::ops::Mul for &Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        let d = self.get_data() * rhs.get_data();
        let mut v = Value_::new(d);
        v.op = Op::Mul((*self).clone(), (*rhs).clone());
        Value(Rc::new(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let v1 = Value::new(1.0);
        assert_eq!(v1.get_data(), 1.0);
        let v2 = Value::new(2.0);
        let v3 = &v1 + &v2;
        assert_eq!(v3.get_data(), 3.0);

        let v4 = &v3 - &v2; // 1.0
        let v5 = &v4 * &v3; // 3.0
        assert_eq!(v5.get_data(), 3.0);
        let v6 = v5.tanh();
        assert_eq!(v5.get_data().tanh(), v6.get_data());
    }
}
