import{_ as s,c as i,a as t,b as a,d as l,r,o}from"./app-CMZGPQLm.js";const c={};function d(m,e){const n=r("ChartJS");return o(),i("div",null,[e[0]||(e[0]=t("h1",null,"Normalization Benchmark",-1)),a(n,{id:"chartjs-3",config:"eJyVU9FuwiAUffcrCE8zaZqCtZW9Lb7sA5YsmfGBWlaJWBrAZWbx38dFqjbT6F6g3Hs495zL7c8IIez2ncDPCFfc4AQCNXfcByDpT4pXQll/XuAMJwgTWCheAjSCrXABECIoXjxfBvI3bVbrQB8zscgiS8uCFLMEZWlOKTt+5CxnsUAAV3y1aYzetfVcK22A0DTVE51OE8RYgshkMr4kr7SphbmFpVew77J2a48lMXHoEdfMvHbumhWSkpJ4A36fUBKMsGmZPWCk9NoIo2G5Z+QO9l9G5rytlUBP240aUJ0clekESlGSssLvRVrQ/AFD/NhumhUJmhUPPc1t7B9HYV/6NXjDunNStzCAcWJ5K7ccYj70yZUVcVSNsJ0Hyi8Yd2d2fbxTu0ZeEECHRCPa+iICMG1lZMVOd3ggJ7YZ2xVXYkC1H7JUwtd6cR/C6IGKkHTSKRB3xsNjSNspDjRDNODFtwt/l9zCK9pxL6qXdZIH3RodRr+Upesr",title:"Softmax%20f32%20Performance%20(size%20%3D%20128%20*%20128%20*%20128)",type:"json"}),a(n,{id:"chartjs-6",config:"eJyVk99uwiAUxu99CsKVJk1TSv/Y3S3e7AGWLJnxglpWiVgawGVm8d3HYXTaTKPe0HD48Z3zHU6/Jwhhe+g5fkK4ZhpHEGiYZS4Ah24nWc2lcfslTnCEMIElxStAA2y49YCPoHDxdBnEX5Veb7x8OAlJlqSI04JGqIxpVrmv22YZDeqerNl622q175qFkkqDmm7raZrnEaqqCBFKZ+fKtdIN19fY9AL7Jhq7cSwJB8eBuOTkpbcXfdCYJpAlJvOi8D4SSu7wUbpLpEr9csvHDfYhHwvWNZKj6W4rR1KDoTSr4jJxqUoS51mE8nmc3WOI/XY7TVwT5sVdL3Od/efIf1du9d6w6q1QHQxfmFbWiR2DmAt9MGl4GFPNTe9A8QmjbvV+iPdy34ozAegQb3nXnEUAU0YEVWxVj0flhDZjs2aSj6QOY5Wau1zP9p1rNarCH1phJRR34uExhOklA5kxDTz/sv7PEjt4RTMbihrK+isPujU5Tn4ASe7qWw==",title:"Softmax%20f32%20Performance%20(size%20%3D%20256%20*%20256%20*%20256)",type:"json"}),a(n,{id:"chartjs-9",config:"eJyVk9FuwiAUhu99CsKVJk3TUtvS3S3e7AGWLJnxglpWiVgawGVm8d3HYVRtptHd0HD4+Dn/OaffE4SwPfQcPyFcM40jCDTMMheAQ7eTrObSuP0SJzhCOIWF4BWgATbcesBHULh4vgzir0qvN14+nIRHliTL45JGiOZxXiYRKtKY0Dyoe7Jm622r1b5rFkoqDWq6rackzyNUVRFKs2x2qVwr3XB9iyVX2DfR2I1j03BwHIhrTl56e81HSpyP0vnI4oy61+Y0Tun8AR+lg9OK+OWejzvsv3wsWNdIjqa7rRxJnRpDSRIXULKqjDMwVMRVcd8Q+602SdxdWjzUmdvsH0f+u3Kr94ZVb4XqYPjCtLJO7BjEXOiDScPDmGpuegeKTxh1q/dDvJf7VlwIQIV4y7vmIgKYMiKoYqt6PEonlBmbNZN8JHUYq9TcvfVs37lWoyz8oRVWQnJnHpohTC8ZyIxp4PmX9X+W2EEXzWxIakjrlB5Ua3Kc/ACyp+qF",title:"Softmax%20f32%20Performance%20(size%20%3D%20512%20*%20512%20*%20512)",type:"json"}),e[1]||(e[1]=l(`<h1>Compilation config</h1><div class="language-cargo line-numbers-mode" data-highlighter="prismjs" data-ext="cargo"><pre><code><span class="line">[profile.release]</span>
<span class="line">opt-level = 3</span>
<span class="line">incremental = true</span>
<span class="line">debug = true</span>
<span class="line">lto = &quot;fat&quot;</span>
<span class="line">codegen-units = 1</span>
<span class="line"></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h1>Running Threads</h1><p><code>10</code></p><h1>Device specification</h1><p><code>CPU</code>: 12th Gen Intel(R) Core(TM) i5-12600K 3.69 GHz</p><p><code>RAM</code>: G.SKILL Trident Z Royal Series (Intel XMP) DDR4 64GB</p><p><code>System</code>: Windows 11 Pro 23H2</p>`,8))])}const f=s(c,[["render",d]]),h=JSON.parse('{"path":"/benchmarks/normalization.html","title":"Normalization Benchmark","lang":"zh-CN","frontmatter":{},"git":{"updatedTime":1744898497000,"contributors":[{"name":"Jianqoq","username":"Jianqoq","email":"ljj1849532909@gmail.com","commits":1,"url":"https://github.com/Jianqoq"}],"changelog":[{"hash":"1348819f0a3d8f5d5425af486b45bab02602f888","time":1744898497000,"email":"ljj1849532909@gmail.com","author":"Jianqoq","message":"use f32 as test type"}]},"filePathRelative":"benchmarks/normalization.md"}');export{f as comp,h as data};
