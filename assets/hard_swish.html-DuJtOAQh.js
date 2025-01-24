import{_ as a,c as n,a as e,o as t}from"./app-CPrDDAiS.js";const p={};function l(i,s){return t(),n("div",null,s[0]||(s[0]=[e(`<h1 id="hard-swish" tabindex="-1"><a class="header-anchor" href="#hard-swish"><span>hard_swish</span></a></h1><div class="language-rust line-numbers-mode" data-highlighter="prismjs" data-ext="rs" data-title="rs"><pre><code><span class="line"><span class="token function">hard_swish</span><span class="token punctuation">(</span>x<span class="token punctuation">:</span> <span class="token operator">&amp;</span><span class="token class-name">Tensor</span><span class="token operator">&lt;</span><span class="token class-name">T</span><span class="token operator">&gt;</span><span class="token punctuation">)</span> <span class="token punctuation">-&gt;</span> <span class="token class-name">Result</span><span class="token operator">&lt;</span><span class="token class-name">Tensor</span><span class="token operator">&lt;</span><span class="token class-name">C</span><span class="token operator">&gt;</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token operator">&gt;</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div></div></div><p>Compute <span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mstyle mathsize="1.2em"><mi>x</mi><mo>⋅</mo><mtext>min</mtext><mo stretchy="false">(</mo><mtext>max</mtext><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><mfrac><mi>x</mi><mn>6</mn></mfrac><mo>+</mo><mn>0.5</mn><mo stretchy="false">)</mo><mo separator="true">,</mo><mn>1</mn><mo stretchy="false">)</mo></mstyle></mrow><annotation encoding="application/x-tex">\\large x \\cdot \\text{min}(\\text{max}(0, \\frac{x}{6} + 0.5), 1)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height:0.5333em;"></span><span class="mord mathnormal sizing reset-size6 size7">x</span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin sizing reset-size6 size7">⋅</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:1.314em;vertical-align:-0.414em;"></span><span class="mord text sizing reset-size6 size7"><span class="mord">min</span></span><span class="mopen sizing reset-size6 size7">(</span><span class="mord text sizing reset-size6 size7"><span class="mord">max</span></span><span class="mopen sizing reset-size6 size7">(</span><span class="mord sizing reset-size6 size7">0</span><span class="mpunct sizing reset-size6 size7">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord sizing reset-size6 size7"><span class="mopen nulldelimiter sizing reset-size7 size6"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height:0.681em;"><span style="top:-2.655em;"><span class="pstrut" style="height:3em;"></span><span class="sizing reset-size7 size4 mtight"><span class="mord mtight"><span class="mord mtight">6</span></span></span></span><span style="top:-3.23em;"><span class="pstrut" style="height:3em;"></span><span class="frac-line" style="border-bottom-width:0.04em;"></span></span><span style="top:-3.394em;"><span class="pstrut" style="height:3em;"></span><span class="sizing reset-size7 size4 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">x</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height:0.345em;"><span></span></span></span></span></span><span class="mclose nulldelimiter sizing reset-size7 size6"></span></span><span class="mspace" style="margin-right:0.2222em;"></span><span class="mbin sizing reset-size6 size7">+</span><span class="mspace" style="margin-right:0.2222em;"></span></span><span class="base"><span class="strut" style="height:1.2em;vertical-align:-0.3em;"></span><span class="mord sizing reset-size6 size7">0.5</span><span class="mclose sizing reset-size6 size7">)</span><span class="mpunct sizing reset-size6 size7">,</span><span class="mspace" style="margin-right:0.1667em;"></span><span class="mord sizing reset-size6 size7">1</span><span class="mclose sizing reset-size6 size7">)</span></span></span></span> for all elements. A piece-wise linear approximation of the swish function.</p><h2 id="parameters" tabindex="-1"><a class="header-anchor" href="#parameters"><span>Parameters:</span></a></h2><p><code>x</code>: Input values</p><h2 id="returns" tabindex="-1"><a class="header-anchor" href="#returns"><span>Returns:</span></a></h2><p>Tensor with type <code>C</code></p><h2 id="examples" tabindex="-1"><a class="header-anchor" href="#examples"><span>Examples:</span></a></h2><div class="language-rust line-numbers-mode" data-highlighter="prismjs" data-ext="rs" data-title="rs"><pre><code><span class="line"><span class="token keyword">use</span> <span class="token namespace">tensor_dyn<span class="token punctuation">::</span></span><span class="token punctuation">{</span><span class="token class-name">FloatUnaryOps</span><span class="token punctuation">,</span> <span class="token class-name">Tensor</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token punctuation">}</span><span class="token punctuation">;</span></span>
<span class="line"></span>
<span class="line"><span class="token keyword">fn</span> <span class="token function-definition function">main</span><span class="token punctuation">(</span><span class="token punctuation">)</span> <span class="token punctuation">-&gt;</span> <span class="token class-name">Result</span><span class="token operator">&lt;</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">,</span> <span class="token class-name">TensorError</span><span class="token operator">&gt;</span> <span class="token punctuation">{</span></span>
<span class="line">    <span class="token keyword">let</span> a <span class="token operator">=</span> <span class="token class-name">Tensor</span><span class="token punctuation">::</span><span class="token operator">&lt;</span><span class="token keyword">f32</span><span class="token operator">&gt;</span><span class="token punctuation">::</span><span class="token function">new</span><span class="token punctuation">(</span><span class="token punctuation">[</span><span class="token number">2.0</span><span class="token punctuation">]</span><span class="token punctuation">)</span><span class="token punctuation">;</span></span>
<span class="line">    <span class="token keyword">let</span> b <span class="token operator">=</span> a<span class="token punctuation">.</span><span class="token function">hard_swish</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token operator">?</span><span class="token punctuation">;</span></span>
<span class="line">    <span class="token macro property">println!</span><span class="token punctuation">(</span><span class="token string">&quot;{}&quot;</span><span class="token punctuation">,</span> b<span class="token punctuation">)</span><span class="token punctuation">;</span>  <span class="token comment">// prints: 1.6666666</span></span>
<span class="line">    <span class="token class-name">Ok</span><span class="token punctuation">(</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">)</span></span>
<span class="line"><span class="token punctuation">}</span></span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,9)]))}const o=a(p,[["render",l],["__file","hard_swish.html.vue"]]),r=JSON.parse('{"path":"/user_guide/unary/hard_swish.html","title":"hard_swish","lang":"zh-CN","frontmatter":{},"headers":[{"level":2,"title":"Parameters:","slug":"parameters","link":"#parameters","children":[]},{"level":2,"title":"Returns:","slug":"returns","link":"#returns","children":[]},{"level":2,"title":"Examples:","slug":"examples","link":"#examples","children":[]}],"git":{"updatedTime":1737695822000,"contributors":[{"name":"Jianqoq","username":"Jianqoq","email":"120760306+Jianqoq@users.noreply.github.com","commits":1,"url":"https://github.com/Jianqoq"}]},"filePathRelative":"user_guide/unary/hard_swish.md"}');export{o as comp,r as data};
