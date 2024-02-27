use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

struct Value(Rc<Value_>);

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
    todo!()
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
}

#[derive(Debug)]
enum Op {
    None,
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}
