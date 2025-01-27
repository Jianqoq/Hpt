### How to add new operator

steps:

1. Identify what trait the operator belongs to. All the existing operator traits are [here](https://github.com/Jianqoq/Hpt/tree/d9a51874b3447d562b7c9d043b50eb05259b78c4/tensor-traits/src/ops), if the operator doesn't belong to any of the traits, create a new one.

2. Implement the trait method at [here](https://github.com/Jianqoq/Hpt/tree/main/tensor-dyn/src/ops) based on what backend you want to implement, mostly, you should implement for all backends.

3. Ensure performance is ideal by comparing with other frameworks

4. Write test cases at [here](https://github.com/Jianqoq/Hpt/tree/main/tensor-tests/src/tensor_dyn). Make sure to follow the Dev Guide test cases rules.

5. commit and make a pull request