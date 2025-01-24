import{_ as n,c as a,a as e,o as p}from"./app-CPrDDAiS.js";const t={};function o(l,s){return p(),a("div",null,s[0]||(s[0]=[e(`<h1 id="zipf-like" tabindex="-1"><a class="header-anchor" href="#zipf-like"><span>zipf_like</span></a></h1><div class="language-rust line-numbers-mode" data-highlighter="prismjs" data-ext="rs" data-title="rs"><pre><code><span class="line"><span class="token function">zipf_like</span><span class="token punctuation">(</span></span>
<span class="line">    x<span class="token punctuation">:</span> <span class="token operator">&amp;</span><span class="token class-name">Tensor</span><span class="token operator">&lt;</span><span class="token class-name">T</span><span class="token operator">&gt;</span><span class="token punctuation">,</span></span>
<span class="line">    n<span class="token punctuation">:</span> <span class="token keyword">u64</span><span class="token punctuation">,</span></span>
<span class="line">    s<span class="token punctuation">:</span> <span class="token class-name">T</span></span>
<span class="line"><span class="token punctuation">)</span> <span class="token punctuation">-&gt;</span> <span class="token class-name">Result</span><span class="token operator">&lt;</span><span class="token class-name">Tensor</span><span class="token operator">&lt;</span><span class="token class-name">T</span><span class="token operator">&gt;</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token operator">&gt;</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>Same as <code>zipf</code> but the shape will be based on <code>x</code>. Creates a Tensor with values drawn from a Zipf distribution with specified number of elements and exponent parameter.</p><h2 id="parameters" tabindex="-1"><a class="header-anchor" href="#parameters"><span>Parameters:</span></a></h2><p><code>x</code>: Input Tensor to derive the shape from</p><p><code>n</code>: Number of elements (N). Defines the range of possible values [1, N].</p><p><code>s</code>: Exponent parameter (s). Controls the skewness of the distribution. Must be greater than 1.</p><h2 id="returns" tabindex="-1"><a class="header-anchor" href="#returns"><span>Returns:</span></a></h2><p>Tensor with type <code>T</code> containing random values from the Zipf distribution.</p><h2 id="examples" tabindex="-1"><a class="header-anchor" href="#examples"><span>Examples:</span></a></h2><div class="language-rust line-numbers-mode" data-highlighter="prismjs" data-ext="rs" data-title="rs"><pre><code><span class="line"><span class="token keyword">use</span> <span class="token namespace">tensor_dyn<span class="token punctuation">::</span></span><span class="token punctuation">{</span><span class="token class-name">Random</span><span class="token punctuation">,</span> <span class="token class-name">Tensor</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token punctuation">}</span><span class="token punctuation">;</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">fn</span> <span class="token function-definition function">main</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token punctuation">-&gt;</span> <span class="token class-name">Result</span><span class="token operator">&lt;</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token operator">&gt;</span> <span class="token punctuation">{</span></span>
<span class="line">    <span class="token comment">// Create an initial tensor</span></span>
<span class="line">    <span class="token keyword">let</span> x <span class="token operator">=</span> <span class="token class-name">Tensor</span><span class="token punctuation">::</span><span class="token operator">&lt;</span><span class="token keyword">f32</span><span class="token operator">&gt;</span><span class="token punctuation">::</span><span class="token function">randn</span><span class="token punctuation">(</span><span class="token operator">&amp;</span><span class="token punctuation">[</span><span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">10</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token operator">?</span><span class="token punctuation">;</span></span>
<span class="line">    </span>
<span class="line">    <span class="token comment">// Create a new tensor with same shape as x but with Zipf distribution</span></span>
<span class="line">    <span class="token keyword">let</span> z <span class="token operator">=</span> x<span class="token punctuation">.</span><span class="token function">zipf_like</span><span class="token punctuation">(</span><span class="token number">1000</span><span class="token punctuation">,</span> <span class="token number">2.0</span><span class="token punctuation">)</span><span class="token operator">?</span><span class="token punctuation">;</span></span>
<span class="line">    <span class="token macro property">println!</span><span class="token punctuation">(</span><span class="token string">&quot;{}&quot;</span><span class="token punctuation">,</span> z<span class="token punctuation">)</span><span class="token punctuation">;</span></span>
<span class="line">    <span class="token class-name">Ok</span><span class="token punctuation">(</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token punctuation">}</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,11)]))}const i=n(t,[["render",o],["__file","zipf_like.html.vue"]]),r=JSON.parse('{"path":"/user_guide/random/zipf_like.html","title":"zipf_like","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"Parameters:","slug":"parameters","link":"#parameters","children":[]},{"level":2,"title":"Returns:","slug":"returns","link":"#returns","children":[]},{"level":2,"title":"Examples:","slug":"examples","link":"#examples","children":[]}],"git":{"updatedTime":1737695822000,"contributors":[{"name":"Jianqoq","username":"Jianqoq","email":"120760306+Jianqoq@users.noreply.github.com","commits":1,"url":"https://github.com/Jianqoq"}]},"filePathRelative":"user_guide/random/zipf_like.md"}');export{i as comp,r as data};
