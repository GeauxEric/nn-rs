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
