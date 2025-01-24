import{_ as n,c as a,a as e,o as t}from"./app-CPrDDAiS.js";const p={};function o(l,s){return t(),a("div",null,s[0]||(s[0]=[e(`<h1 id="gumbel" tabindex="-1"><a class="header-anchor" href="#gumbel"><span>gumbel</span></a></h1><div class="language-rust line-numbers-mode" data-highlighter="prismjs" data-ext="rs" data-title="rs"><pre><code><span class="line"><span class="token function">gumbel</span><span class="token punctuation">(</span></span>
<span class="line">    mu<span class="token punctuation">:</span> <span class="token class-name">T</span><span class="token punctuation">,</span></span>
<span class="line">    beta<span class="token punctuation">:</span> <span class="token class-name">T</span><span class="token punctuation">,</span></span>
<span class="line">    shape<span class="token punctuation">:</span> <span class="token operator">&amp;</span><span class="token punctuation">[</span><span class="token keyword">i64</span><span class="token punctuation">]</span> <span class="token operator">|</span> <span class="token operator">&amp;</span><span class="token class-name">Vec</span><span class="token operator">&lt;</span><span class="token keyword">i64</span><span class="token operator">&gt;</span> <span class="token operator">|</span> <span class="token operator">&amp;</span><span class="token punctuation">[</span><span class="token keyword">i64</span><span class="token punctuation">;</span> _<span class="token punctuation">]</span></span>
<span class="line"><span class="token punctuation">)</span> <span class="token punctuation">-&gt;</span> <span class="token class-name">Result</span><span class="token operator">&lt;</span><span class="token class-name">Tensor</span><span class="token operator">&lt;</span><span class="token class-name">T</span><span class="token operator">&gt;</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token operator">&gt;</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p>Create a Tensor with values drawn from a Gumbel distribution (also known as the Extreme Value Type I distribution) with location parameter <code>mu</code> and scale parameter <code>beta</code>. The Gumbel distribution is commonly used to model the distribution of extreme values.</p><h2 id="parameters" tabindex="-1"><a class="header-anchor" href="#parameters"><span>Parameters:</span></a></h2><p><code>mu</code>: Location parameter (μ) of the Gumbel distribution.</p><p><code>beta</code>: Scale parameter (β) of the Gumbel distribution. Must be positive.</p><p><code>shape</code>: Shape of the output tensor.</p><h2 id="returns" tabindex="-1"><a class="header-anchor" href="#returns"><span>Returns:</span></a></h2><p>Tensor with type <code>T</code> containing random values from the Gumbel distribution.</p><h2 id="examples" tabindex="-1"><a class="header-anchor" href="#examples"><span>Examples:</span></a></h2><div class="language-rust line-numbers-mode" data-highlighter="prismjs" data-ext="rs" data-title="rs"><pre><code><span class="line"><span class="token keyword">use</span> <span class="token namespace">tensor_dyn<span class="token punctuation">::</span></span><span class="token punctuation">{</span><span class="token class-name">Random</span><span class="token punctuation">,</span> <span class="token class-name">Tensor</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token punctuation">}</span><span class="token punctuation">;</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">fn</span> <span class="token function-definition function">main</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token punctuation">-&gt;</span> <span class="token class-name">Result</span><span class="token operator">&lt;</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token operator">&gt;</span> <span class="token punctuation">{</span></span>
<span class="line">    <span class="token comment">// Create a 10x10 tensor with Gumbel distribution (μ=0.0, β=1.0)</span></span>
<span class="line">    <span class="token keyword">let</span> a <span class="token operator">=</span> <span class="token class-name">Tensor</span><span class="token punctuation">::</span><span class="token operator">&lt;</span><span class="token keyword">f32</span><span class="token operator">&gt;</span><span class="token punctuation">::</span><span class="token function">gumbel</span><span class="token punctuation">(</span><span class="token number">0.0</span><span class="token punctuation">,</span> <span class="token number">1.0</span><span class="token punctuation">,</span> <span class="token operator">&amp;</span><span class="token punctuation">[</span><span class="token number">10</span><span class="token punctuation">,</span> <span class="token number">10</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token operator">?</span><span class="token punctuation">;</span></span>
<span class="line">    <span class="token macro property">println!</span><span class="token punctuation">(</span><span class="token string">&quot;{}&quot;</span><span class="token punctuation">,</span> a<span class="token punctuation">)</span><span class="token punctuation">;</span></span>
<span class="line">    <span class="token class-name">Ok</span><span class="token punctuation">(</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token punctuation">}</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,11)]))}const i=n(p,[["render",o],["__file","gumbel.html.vue"]]),r=JSON.parse('{"path":"/user_guide/random/gumbel.html","title":"gumbel","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"Parameters:","slug":"parameters","link":"#parameters","children":[]},{"level":2,"title":"Returns:","slug":"returns","link":"#returns","children":[]},{"level":2,"title":"Examples:","slug":"examples","link":"#examples","children":[]}],"git":{"updatedTime":1737695822000,"contributors":[{"name":"Jianqoq","username":"Jianqoq","email":"120760306+Jianqoq@users.noreply.github.com","commits":1,"url":"https://github.com/Jianqoq"}]},"filePathRelative":"user_guide/random/gumbel.md"}');export{i as comp,r as data};
