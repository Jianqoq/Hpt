use serde::ser::SerializeStruct;
use std::{ alloc::Layout, fmt::{ Debug, Display, Formatter }, ops::{ Deref, DerefMut, Index, IndexMut } };

use serde::Serialize;

/// Pointer wrapper struct for raw pointers
/// This is for wrapping raw pointers to make them safe for multithreading
///
/// This is for internal use only
#[derive(Debug, Copy, Clone)]
pub struct Pointer<T> {
    pub ptr: *mut T,
}

impl<T> Pointer<T> {
    /// return raw pointer
    ///
    /// # Returns
    /// `*mut T`
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = 10;
    /// let a = Pointer::<i32>::new(_a as *mut i32);
    /// let b = a.get_ptr();
    /// assert_eq!(b, _a as *mut i32);
    /// ```
    #[inline(always)]
    pub fn get_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Wrap a raw pointer into a Pointer struct for supporting `Send` in multithreading, zero cost
    ///
    /// # Arguments
    /// `ptr` - `*mut T`
    ///
    /// # Returns
    /// `Pointer<T>`
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = 10i32;
    /// let a = Pointer::<i32>::new(_a as *mut i32);
    /// assert_eq!(a.read(), 10);
    /// ```
    #[inline(always)]
    pub fn new(ptr: *mut T) -> Self {
        Self { ptr }
    }

    /// return the address of the pointer
    ///
    /// # Returns
    /// `usize`
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = 10i32;
    /// let a = Pointer::<i32>::new(_a as *mut i32);
    /// let b = a.address();
    /// assert_eq!(b, _a as usize);
    /// ```
    pub fn address(&self) -> usize {
        self.ptr as usize
    }

    /// read the value of the pointer in the current address
    ///
    /// # Returns
    /// `T`
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = unsafe { std::alloc::alloc(std::alloc::Layout::new::<i32>()) as *mut i32 };
    /// let mut a = Pointer::<i32>::new(_a);
    /// unsafe { _a.write(10); }
    /// assert_eq!(a.read(), 10);
    /// unsafe { std::alloc::dealloc(_a as *mut u8, std::alloc::Layout::new::<i32>()); }
    /// ```
    #[inline(always)]
    pub fn read(&self) -> T {
        unsafe { self.ptr.read() }
    }

    /// modify the value of the pointer in the address by the specified offset
    ///
    /// # Arguments
    /// `offset` - the offset from the current address
    /// `value` - the value to be written
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = unsafe { std::alloc::alloc(std::alloc::Layout::new::<i32>()) as *mut i32 };
    /// let mut a = Pointer::<i32>::new(_a);
    /// a.modify(0, 10);
    /// assert_eq!(a.read(), 10);
    /// unsafe { std::alloc::dealloc(_a as *mut u8, std::alloc::Layout::new::<i32>()); }
    /// ```
    #[inline(always)]
    pub fn modify(&mut self, offset: i64, value: T) {
        unsafe {
            self.ptr.offset(offset as isize).write(value);
        }
    }

    /// write the value of the pointer in the current address
    ///
    /// # Arguments
    /// `value` - the value to be written
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = unsafe { std::alloc::alloc(std::alloc::Layout::new::<i32>()) as *mut i32 };
    /// let mut a = Pointer::<i32>::new(_a);
    /// a.write(10);
    /// assert_eq!(a.read(), 10);
    /// unsafe { std::alloc::dealloc(_a as *mut u8, std::alloc::Layout::new::<i32>()); }
    /// ```
    #[inline(always)]
    pub fn write(&mut self, value: T) {
        unsafe {
            self.ptr.write(value);
        }
    }

    /// inplace increment the value of the pointer in the current address
    ///
    /// # Arguments
    /// `value` - the value to be added
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = unsafe { std::alloc::alloc(std::alloc::Layout::new::<i32>()) as *mut i32 };
    /// unsafe { _a.write(10); }
    /// let mut a = Pointer::<i32>::new(_a);
    /// a.add(0);
    /// assert_eq!(a.read(), 10);
    /// unsafe { std::alloc::dealloc(_a as *mut u8, std::alloc::Layout::new::<i32>()); }
    /// ```
    #[inline(always)]
    pub fn add(&mut self, offset: usize) {
        unsafe {
            self.ptr = self.ptr.add(offset);
        }
    }

    /// inplace decrement the value of the pointer in the current address
    ///
    /// # Arguments
    /// `value` - the value to be subtracted
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = unsafe { std::alloc::alloc(std::alloc::Layout::new::<i32>()) as *mut i32 };
    /// unsafe { _a.write(10); }
    /// let mut a = Pointer::<i32>::new(_a);
    /// a.sub(0);
    /// assert_eq!(a.read(), 10);
    /// unsafe { std::alloc::dealloc(_a as *mut u8, std::alloc::Layout::new::<i32>()); }
    /// ```
    #[inline(always)]
    pub fn sub(&mut self, offset: usize) {
        unsafe {
            self.ptr = self.ptr.sub(offset);
        }
    }

    /// inplace offset the value of the pointer in the current address
    ///
    /// # Arguments
    /// `offset` - the offset to be added
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = unsafe { std::alloc::alloc(std::alloc::Layout::new::<i32>()) as *mut i32 };
    /// unsafe { _a.write(10); }
    /// let mut a = Pointer::<i32>::new(_a);
    /// a.offset(0);
    /// assert_eq!(a.read(), 10);
    /// unsafe { std::alloc::dealloc(_a as *mut u8, std::alloc::Layout::new::<i32>()); }
    /// ```
    #[inline(always)]
    pub fn offset(&mut self, offset: i64) {
        unsafe {
            self.ptr = self.ptr.offset(offset as isize);
        }
    }

    /// inplace jump the value of the pointer in the current address
    ///
    /// # Arguments
    /// `offset` - the offset to be added
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = unsafe { std::alloc::alloc(std::alloc::Layout::new::<i32>()) as *mut i32 };
    /// unsafe { _a.write(10); }
    /// let mut a = Pointer::<i32>::new(_a);
    /// a.jump(0);
    /// assert_eq!(a.read(), 10);
    /// unsafe { std::alloc::dealloc(_a as *mut u8, std::alloc::Layout::new::<i32>()); }
    /// ```
    #[inline(always)]
    pub fn jump(&mut self, offset: usize) -> *mut T {
        unsafe { self.ptr.add(offset) }
    }
}

unsafe impl<T> Send for Pointer<T> {}

impl<T: Display> Index<i64> for Pointer<T> {
    type Output = T;
    fn index(&self, index: i64) -> &Self::Output {
        unsafe { &*self.ptr.offset(index as isize) }
    }
}

impl<T: Display> Index<isize> for Pointer<T> {
    type Output = T;
    fn index(&self, index: isize) -> &Self::Output {
        unsafe { &*self.ptr.offset(index) }
    }
}

impl<T: Display> IndexMut<i64> for Pointer<T> {
    fn index_mut(&mut self, index: i64) -> &mut Self::Output {
        unsafe { &mut *self.ptr.offset(index as isize) }
    }
}

impl<T: Display> IndexMut<isize> for Pointer<T> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        unsafe { &mut *self.ptr.offset(index) }
    }
}

impl<T> Deref for Pointer<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr }
    }
}

impl<T> DerefMut for Pointer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.ptr }
    }
}

unsafe impl<T> Sync for Pointer<T> {}

impl<T: Display> Display for Pointer<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Pointer( ptr: {}, val: {} )", self.ptr as usize, unsafe { self.ptr.read() })
    }
}

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub struct VoidPointer {
    pub ptr: *mut u8,
    pub layout: Layout,
}

impl Serialize for VoidPointer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: serde::Serializer {
        let mut state = serializer.serialize_struct("VoidPointer", 2)?;
        state.serialize_field("ptr", &(self.ptr as usize))?;
        state.serialize_field("align", &self.layout.align())?;
        state.end()
    }
}

impl VoidPointer {
    /// return raw pointer
    ///
    /// # Returns
    /// `*mut T`
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = 10;
    /// let a = Pointer::<i32>::new(_a as *mut i32);
    /// let b = a.get_ptr();
    /// assert_eq!(b, _a as *mut i32);
    /// ```
    #[inline(always)]
    pub fn get_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Wrap a raw pointer into a Pointer struct for supporting `Send` in multithreading, zero cost
    ///
    /// # Arguments
    /// `ptr` - `*mut T`
    ///
    /// # Returns
    /// `Pointer<T>`
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = 10i32;
    /// let a = Pointer::<i32>::new(_a as *mut i32);
    /// assert_eq!(a.read(), 10);
    /// ```
    #[inline(always)]
    pub fn new(ptr: *mut u8, layout: Layout) -> Self {
        Self { ptr, layout }
    }

    /// return the address of the pointer
    ///
    /// # Returns
    /// `usize`
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = 10i32;
    /// let a = Pointer::<i32>::new(_a as *mut i32);
    /// let b = a.address();
    /// assert_eq!(b, _a as usize);
    /// ```
    pub fn address(&self) -> usize {
        self.ptr as usize
    }

    /// inplace increment the value of the pointer in the current address
    ///
    /// # Arguments
    /// `value` - the value to be added
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = unsafe { std::alloc::alloc(std::alloc::Layout::new::<i32>()) as *mut i32 };
    /// unsafe { _a.write(10); }
    /// let mut a = Pointer::<i32>::new(_a);
    /// a.add(0);
    /// assert_eq!(a.read(), 10);
    /// unsafe { std::alloc::dealloc(_a as *mut u8, std::alloc::Layout::new::<i32>()); }
    /// ```
    #[inline(always)]
    pub fn add(&mut self, offset: usize, elsize: usize) {
        unsafe {
            self.ptr = self.ptr.add(offset * elsize);
        }
    }

    /// inplace decrement the value of the pointer in the current address
    ///
    /// # Arguments
    /// `value` - the value to be subtracted
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = unsafe { std::alloc::alloc(std::alloc::Layout::new::<i32>()) as *mut i32 };
    /// unsafe { _a.write(10); }
    /// let mut a = Pointer::<i32>::new(_a);
    /// a.sub(0);
    /// assert_eq!(a.read(), 10);
    /// unsafe { std::alloc::dealloc(_a as *mut u8, std::alloc::Layout::new::<i32>()); }
    /// ```
    #[inline(always)]
    pub fn sub(&mut self, offset: usize, elsize: usize) {
        unsafe {
            self.ptr = self.ptr.sub(offset * elsize);
        }
    }

    /// inplace offset the value of the pointer in the current address
    ///
    /// # Arguments
    /// `offset` - the offset to be added
    ///
    /// # Example
    /// ```
    /// use tensor_pointer::Pointer;
    /// let mut _a = unsafe { std::alloc::alloc(std::alloc::Layout::new::<i32>()) as *mut i32 };
    /// unsafe { _a.write(10); }
    /// let mut a = Pointer::<i32>::new(_a);
    /// a.offset(0);
    /// assert_eq!(a.read(), 10);
    /// unsafe { std::alloc::dealloc(_a as *mut u8, std::alloc::Layout::new::<i32>()); }
    /// ```
    #[inline(always)]
    pub fn offset(&mut self, offset: i64, elsize: i64) {
        unsafe {
            self.ptr = self.ptr.offset((offset * elsize) as isize);
        }
    }
}

unsafe impl Send for VoidPointer {}

impl Debug for VoidPointer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VoidPointer( ptr: {}, layout: {{ size: {} }} )",
            self.ptr as usize,
            self.layout.size()
        )
    }
}

unsafe impl Sync for VoidPointer {}
