use std::{
    fmt::{Debug, Display, Formatter},
    ops::{Add, AddAssign, Deref, DerefMut, Index, IndexMut, SubAssign},
};

/// Pointer wrapper struct for raw pointers
/// This is for wrapping raw pointers to make them safe for multithreading
///
/// This is for internal use only
#[derive(Debug, Copy, Clone)]
pub struct Pointer<T> {
    /// raw pointer
    pub ptr: *mut T,
    /// len of the pointer, it is used when the `bound_check` feature is enabled
    #[cfg(feature = "bound_check")]
    pub len: i64,
}

impl<T> Pointer<T> {
    /// return a slice of the pointer
    ///
    /// # Returns
    /// `&[T]`
    #[cfg(feature = "bound_check")]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len as usize) }
    }
    /// cast the pointer to a new type
    ///
    /// # Arguments
    /// `U` - the new type
    ///
    /// # Returns
    /// `Pointer<U>`
    pub fn cast<U>(&self) -> Pointer<U> {
        #[cfg(feature = "bound_check")]
        return Pointer::new(self.ptr as *mut U, self.len);
        #[cfg(not(feature = "bound_check"))]
        return Pointer::new(self.ptr as *mut U);
    }
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
    #[cfg(not(feature = "bound_check"))]
    #[inline(always)]
    pub fn new(ptr: *mut T) -> Self {
        Self { ptr }
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
    #[cfg(feature = "bound_check")]
    #[inline(always)]
    pub fn new(ptr: *mut T, len: i64) -> Self {
        Self { ptr, len }
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
}

unsafe impl<T> Send for Pointer<T> {}

impl<T> Index<i64> for Pointer<T> {
    type Output = T;
    fn index(&self, index: i64) -> &Self::Output {
        #[cfg(feature = "bound_check")]
        {
            if index < 0 || index >= (self.len as i64) {
                panic!("index out of bounds. index: {}, len: {}", index, self.len);
            }
        }
        unsafe { &*self.ptr.offset(index as isize) }
    }
}

impl<T: Display> Index<isize> for Pointer<T> {
    type Output = T;
    fn index(&self, index: isize) -> &Self::Output {
        #[cfg(feature = "bound_check")]
        {
            if index < 0 || (index as i64) >= (self.len as i64) {
                panic!("index out of bounds. index: {}, len: {}", index, self.len);
            }
        }
        unsafe { &*self.ptr.offset(index) }
    }
}

impl<T: Display> Index<usize> for Pointer<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        #[cfg(feature = "bound_check")]
        {
            if (index as i64) >= (self.len as i64) {
                panic!("index out of bounds. index: {}, len: {}", index, self.len);
            }
        }
        unsafe { &*self.ptr.add(index) }
    }
}

impl<T: Display> IndexMut<i64> for Pointer<T> {
    fn index_mut(&mut self, index: i64) -> &mut Self::Output {
        #[cfg(feature = "bound_check")]
        {
            if index < 0 || index >= (self.len as i64) {
                panic!("index out of bounds. index: {}, len: {}", index, self.len);
            }
        }
        unsafe { &mut *self.ptr.offset(index as isize) }
    }
}

impl<T: Display> IndexMut<isize> for Pointer<T> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        #[cfg(feature = "bound_check")]
        {
            if index < 0 || (index as i64) >= (self.len as i64) {
                panic!("index out of bounds. index: {}, len: {}", index, self.len);
            }
        }
        unsafe { &mut *self.ptr.offset(index) }
    }
}

impl<T: Display> IndexMut<usize> for Pointer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        #[cfg(feature = "bound_check")]
        {
            if (index as i64) >= (self.len as i64) {
                panic!("index out of bounds. index: {}, len: {}", index, self.len);
            }
        }
        unsafe { &mut *self.ptr.add(index) }
    }
}

impl<T> AddAssign<usize> for Pointer<T> {
    fn add_assign(&mut self, rhs: usize) {
        #[cfg(feature = "bound_check")]
        {
            self.len -= rhs as i64;
            assert!(self.len >= 0);
        }
        unsafe {
            self.ptr = self.ptr.add(rhs);
        }
    }
}

impl<T> Add<usize> for Pointer<T> {
    type Output = Self;
    fn add(self, rhs: usize) -> Self::Output {
        #[cfg(feature = "bound_check")]
        unsafe {
            Self {
                ptr: self.ptr.add(rhs),
                len: self.len,
            }
        }
        #[cfg(not(feature = "bound_check"))]
        unsafe {
            Self {
                ptr: self.ptr.add(rhs),
            }
        }
    }
}

impl<T> AddAssign<usize> for &mut Pointer<T> {
    fn add_assign(&mut self, rhs: usize) {
        #[cfg(feature = "bound_check")]
        {
            self.len -= rhs as i64;
        }
        unsafe {
            self.ptr = self.ptr.add(rhs);
        }
    }
}

impl<T> AddAssign<i64> for &mut Pointer<T> {
    fn add_assign(&mut self, rhs: i64) {
        #[cfg(feature = "bound_check")]
        {
            self.len -= rhs;
            assert!(self.len >= 0);
        }
        unsafe {
            self.ptr = self.ptr.offset(rhs as isize);
        }
    }
}

impl<T> Add<usize> for &mut Pointer<T> {
    type Output = Pointer<T>;
    fn add(self, rhs: usize) -> Self::Output {
        #[cfg(feature = "bound_check")]
        unsafe {
            Pointer::new(self.ptr.add(rhs), self.len)
        }
        #[cfg(not(feature = "bound_check"))]
        unsafe {
            Pointer::new(self.ptr.add(rhs))
        }
    }
}

impl<T> AddAssign<isize> for Pointer<T> {
    fn add_assign(&mut self, rhs: isize) {
        #[cfg(feature = "bound_check")]
        {
            self.len -= rhs as i64;
            assert!(self.len >= 0);
        }
        unsafe {
            self.ptr = self.ptr.offset(rhs);
        }
    }
}

impl<T> Add<isize> for Pointer<T> {
    type Output = Self;
    fn add(self, rhs: isize) -> Self::Output {
        #[cfg(feature = "bound_check")]
        unsafe {
            Self {
                ptr: self.ptr.offset(rhs),
                len: self.len,
            }
        }
        #[cfg(not(feature = "bound_check"))]
        unsafe {
            Self {
                ptr: self.ptr.offset(rhs),
            }
        }
    }
}

impl<T> AddAssign<i64> for Pointer<T> {
    fn add_assign(&mut self, rhs: i64) {
        #[cfg(feature = "bound_check")]
        {
            assert!(self.len >= 0);
            self.len -= rhs as i64;
        }
        unsafe {
            self.ptr = self.ptr.offset(rhs as isize);
        }
    }
}

impl<T> Add<i64> for Pointer<T> {
    type Output = Self;
    fn add(self, rhs: i64) -> Self::Output {
        #[cfg(feature = "bound_check")]
        unsafe {
            Self {
                ptr: self.ptr.offset(rhs as isize),
                len: self.len,
            }
        }
        #[cfg(not(feature = "bound_check"))]
        unsafe {
            Self {
                ptr: self.ptr.offset(rhs as isize),
            }
        }
    }
}

impl<T> SubAssign<usize> for Pointer<T> {
    fn sub_assign(&mut self, rhs: usize) {
        #[cfg(feature = "bound_check")]
        {
            self.len += rhs as i64;
        }
        unsafe {
            self.ptr = self.ptr.offset(-(rhs as isize));
        }
    }
}

impl<T> SubAssign<isize> for Pointer<T> {
    fn sub_assign(&mut self, rhs: isize) {
        #[cfg(feature = "bound_check")]
        {
            self.len += rhs as i64;
        }
        unsafe {
            self.ptr = self.ptr.offset(-rhs);
        }
    }
}

impl<T> SubAssign<i64> for Pointer<T> {
    fn sub_assign(&mut self, rhs: i64) {
        #[cfg(feature = "bound_check")]
        {
            self.len += rhs as i64;
        }
        unsafe {
            self.ptr = self.ptr.offset(-rhs as isize);
        }
    }
}

impl<T> SubAssign<i64> for &mut Pointer<T> {
    fn sub_assign(&mut self, rhs: i64) {
        #[cfg(feature = "bound_check")]
        {
            self.len += rhs as i64;
        }
        unsafe {
            self.ptr = self.ptr.offset(-rhs as isize);
        }
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
        write!(
            f,
            "Pointer( ptr: {}, val: {} )",
            self.ptr as usize,
            unsafe { self.ptr.read() }
        )
    }
}
