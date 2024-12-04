#[derive(Debug, Clone, Copy)]
pub(crate) enum ExprType {
    /// A slice literal expression: `[a, b, c, d]`.
    Array,

    /// An assignment expression: `a = compute()`.
    Assign,

    /// An async block: `async { ... }`.
    Async,

    /// An await expression: `fut.await`.
    Await,

    /// A binary operation: `a + b`, `a += b`.
    Binary,

    /// A blocked scope: `{ ... }`.
    Block,

    /// A `break`, with an optional label to break and an optional
    /// expression.
    Break,

    /// A function call expression: `invoke(a, b)`.
    Call,

    /// A cast expression: `foo as f64`.
    Cast,

    /// A closure expression: `|a, b| a + b`.
    Closure,

    /// A const block: `const { ... }`.
    Const,

    /// A `continue`, with an optional label.
    Continue,

    /// Access of a named struct field (`obj.k`) or unnamed tuple struct
    /// field (`obj.0`).
    Field,

    /// A for loop: `for pat in expr { ... }`.
    ForLoop,

    /// An expression contained within invisible delimiters.
    ///
    /// This variant is important for faithfully representing the precedence
    /// of expressions and is related to `None`-delimited spans in a
    /// `TokenStream`.
    Group,

    /// An `if` expression with an optional `else` block: `if expr { ... }
    /// else { ... }`.
    ///
    /// The `else` branch expression may only be an `If` or `Block`
    /// expression, not any of the other types of expression.
    If,

    /// A square bracketed indexing expression: `vector[2]`.
    Index,

    /// The inferred value of a const generic argument, denoted `_`.
    Infer,

    /// A `let` guard: `let Some(x) = opt`.
    Let,

    /// A literal in place of an expression: `1`, `"foo"`.
    Lit,

    /// Conditionless loop: `loop { ... }`.
    Loop,

    /// A macro invocation expression: `format!("{}", q)`.
    Macro,

    /// A `match` expression: `match n { Some(n) => {}, None => {} }`.
    Match,

    /// A method call expression: `x.foo::<T>(a, b)`.
    MethodCall,

    /// A parenthesized expression: `(a + b)`.
    Paren,

    /// A path like `std::mem::replace` possibly containing generic
    /// parameters and a qualified self-type.
    ///
    /// A plain identifier like `x` is a path of length 1.
    Path,

    /// A range expression: `1..2`, `1..`, `..2`, `1..=2`, `..=2`.
    Range,

    /// Address-of operation: `&raw const place` or `&raw mut place`.
    RawAddr,

    /// A referencing operation: `&a` or `&mut a`.
    Reference,

    /// An array literal constructed from one repeated element: `[0u8; N]`.
    Repeat,

    /// A `return`, with an optional value to be returned.
    Return,

    /// A struct literal expression: `Point { x: 1, y: 1 }`.
    ///
    /// The `rest` provides the value of the remaining fields as in `S { a:
    /// 1, b: 1, ..rest }`.
    Struct,

    /// A try-expression: `expr?`.
    Try,

    /// A try block: `try { ... }`.
    TryBlock,

    /// A tuple expression: `(a, b, c, d)`.
    Tuple,

    /// A unary operation: `!x`, `*x`.
    Unary,

    /// An unsafe block: `unsafe { ... }`.
    Unsafe,

    /// Tokens in expression position not interpreted by Syn.
    Verbatim,

    /// A while loop: `while expr { ... }`.
    While,

    /// A yield expression: `yield expr`.
    Yield,

    // For testing exhaustiveness in downstream code, use the following idiom:
    //
    //     match expr {
    //         #![cfg_attr(test, deny(non_exhaustive_omitted_patterns))]
    //
    //         Expr::Array(expr) => {...}
    //         Expr::Assign(expr) => {...}
    //         ...
    //         Expr::Yield(expr) => {...}
    //
    //         _ => { /* some sane fallback */ }
    //     }
    //
    // This way we fail your tests but don't break your library when adding
    // a variant. You will be notified by a test failure when a variant is
    // added, so that you can add code to handle it, but your library will
    // continue to compile and work for downstream users in the interim.
}

impl From<&syn::Expr> for ExprType {
    fn from(expr: &syn::Expr) -> Self {
        match expr {
            syn::Expr::Array(_) => Self::Array,
            syn::Expr::Assign(_) => Self::Assign,
            syn::Expr::Async(_) => Self::Async,
            syn::Expr::Await(_) => Self::Await,
            syn::Expr::Binary(_) => Self::Binary,
            syn::Expr::Block(_) => Self::Block,
            syn::Expr::Break(_) => Self::Break,
            syn::Expr::Call(_) => Self::Call,
            syn::Expr::Cast(_) => Self::Cast,
            syn::Expr::Closure(_) => Self::Closure,
            syn::Expr::Const(_) => Self::Const,
            syn::Expr::Continue(_) => Self::Continue,
            syn::Expr::Field(_) => Self::Field,
            syn::Expr::ForLoop(_) => Self::ForLoop,
            syn::Expr::Group(_) => Self::Group,
            syn::Expr::If(_) => Self::If,
            syn::Expr::Index(_) => Self::Index,
            syn::Expr::Infer(_) => Self::Infer,
            syn::Expr::Let(_) => Self::Let,
            syn::Expr::Lit(_) => Self::Lit,
            syn::Expr::Loop(_) => Self::Loop,
            syn::Expr::Macro(_) => Self::Macro,
            syn::Expr::Match(_) => Self::Match,
            syn::Expr::MethodCall(_) => Self::MethodCall,
            syn::Expr::Paren(_) => Self::Paren,
            syn::Expr::Path(_) => Self::Path,
            syn::Expr::Range(_) => Self::Range,
            syn::Expr::RawAddr(_) => Self::RawAddr,
            syn::Expr::Reference(_) => Self::Reference,
            syn::Expr::Repeat(_) => Self::Repeat,
            syn::Expr::Return(_) => Self::Return,
            syn::Expr::Struct(_) => Self::Struct,
            syn::Expr::Try(_) => Self::Try,
            syn::Expr::TryBlock(_) => Self::TryBlock,
            syn::Expr::Tuple(_) => Self::Tuple,
            syn::Expr::Unary(_) => Self::Unary,
            syn::Expr::Unsafe(_) => Self::Unsafe,
            syn::Expr::Verbatim(_) => Self::Verbatim,
            syn::Expr::While(_) => Self::While,
            syn::Expr::Yield(_) => Self::Yield,
            _ => todo!(),
        }
    }
}

impl From<&mut syn::Expr> for ExprType {
    fn from(expr: &mut syn::Expr) -> Self {
        match expr {
            syn::Expr::Array(_) => Self::Array,
            syn::Expr::Assign(_) => Self::Assign,
            syn::Expr::Async(_) => Self::Async,
            syn::Expr::Await(_) => Self::Await,
            syn::Expr::Binary(_) => Self::Binary,
            syn::Expr::Block(_) => Self::Block,
            syn::Expr::Break(_) => Self::Break,
            syn::Expr::Call(_) => Self::Call,
            syn::Expr::Cast(_) => Self::Cast,
            syn::Expr::Closure(_) => Self::Closure,
            syn::Expr::Const(_) => Self::Const,
            syn::Expr::Continue(_) => Self::Continue,
            syn::Expr::Field(_) => Self::Field,
            syn::Expr::ForLoop(_) => Self::ForLoop,
            syn::Expr::Group(_) => Self::Group,
            syn::Expr::If(_) => Self::If,
            syn::Expr::Index(_) => Self::Index,
            syn::Expr::Infer(_) => Self::Infer,
            syn::Expr::Let(_) => Self::Let,
            syn::Expr::Lit(_) => Self::Lit,
            syn::Expr::Loop(_) => Self::Loop,
            syn::Expr::Macro(_) => Self::Macro,
            syn::Expr::Match(_) => Self::Match,
            syn::Expr::MethodCall(_) => Self::MethodCall,
            syn::Expr::Paren(_) => Self::Paren,
            syn::Expr::Path(_) => Self::Path,
            syn::Expr::Range(_) => Self::Range,
            syn::Expr::RawAddr(_) => Self::RawAddr,
            syn::Expr::Reference(_) => Self::Reference,
            syn::Expr::Repeat(_) => Self::Repeat,
            syn::Expr::Return(_) => Self::Return,
            syn::Expr::Struct(_) => Self::Struct,
            syn::Expr::Try(_) => Self::Try,
            syn::Expr::TryBlock(_) => Self::TryBlock,
            syn::Expr::Tuple(_) => Self::Tuple,
            syn::Expr::Unary(_) => Self::Unary,
            syn::Expr::Unsafe(_) => Self::Unsafe,
            syn::Expr::Verbatim(_) => Self::Verbatim,
            syn::Expr::While(_) => Self::While,
            syn::Expr::Yield(_) => Self::Yield,
            _ => todo!(),
        }
    }
}
