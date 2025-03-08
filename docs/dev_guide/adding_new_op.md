# How to add new Tensor operator

### Things to know
Tensor operators are define in `tensor-traits` crate. If you want to implement new Tensor operator, you may need to create a new trait if there is no suitable trait for the new operator.

### How
2. Implement the trait method at [here](https://github.com/Jianqoq/Hpt/tree/main/hpt/src/backends) based on what backend you want to implement, mostly, you should implement for all backends.

3. Ensure performance is ideal by comparing with other frameworks

4. Write test cases at [here](https://github.com/Jianqoq/Hpt/tree/main/hpt-tests/src/hpt). Make sure to follow the Dev Guide test cases rules.

5. commit and make a pull request