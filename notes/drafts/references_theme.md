# Theme references, DPO and licences (draft, fork D, 9 November 2026)

Every item below was opened and checked on 9 November 2026: arXiv API metadata (title, authors, comments, journal reference, DOI) and, where a venue is claimed, the venue's own page or the Crossref record of the DOI. Already in paper/references.bib and not repeated here: Artificial Hivemind, Multilingual Artificial Hivemind, Kirk et al. (RLHF and diversity), Karouzos et al. (where diversity collapses in post-training), Verbalized Sampling, Kalavasis et al.

## 1. References for the special theme (homogenisation and knowledge collapse)

| key | verified as | how we use it (one sentence) |
|---|---|---|
| shumailov2024collapse | Nature 631(8022): 755 to 759, 2024; DOI 10.1038/s41586-024-07566-y (Crossref; nature.com abstract read) | Training on generated text makes "tails of the original content distribution disappear"; truncation makes that tail loss exact and immediate for any text a model writes under a shipped sampler, before any retraining. |
| peterson2025collapse | AI & Society 40(5): 3249 to 3269, 2025; DOI 10.1007/s00146-024-02173-x (Crossref; arXiv 2404.03502 journal reference) | Defines "knowledge collapse": LLMs "naturally generate output towards the 'center' of the distribution". We measure how much of that centring is a decoding setting rather than the model. |
| dohmatob2024tails | ICML 2024, PMLR 235: 11165 to 11197 (proceedings.mlr.press/v235/dohmatob24b.html) | Model collapse as a change of scaling laws once synthetic data enters training; cite with Shumailov et al. as the downstream cost of tail loss. |
| guo2024curious | Findings of NAACL 2024, pp. 3589 to 3604; DOI 10.18653/v1/2024.findings-naacl.228 (ACL Anthology BibTeX) | Recursive training on synthetic text lowers lexical, syntactic and semantic diversity; our reachable set bounds the content such synthetic data can contain. |
| alemohammad2024mad | ICLR 2024 poster (iclr.cc/virtual/2024/poster/18592; author list read there) | Self-consuming loops lose diversity unless fresh real data enters each generation; the paper also varies how far samples are "biased to trade off data quality versus diversity", which is what truncation does. |
| wright2026epistemic | arXiv 2510.04226 v7, arXiv comment "Accepted to EMNLP 2026" (title changed in later versions; use v7's) | Measures epistemic diversity of LLM claims as a knowledge-collapse risk; we add an exact, decoding-side reason why some claims never appear. |
| hodel2025ecosystem | arXiv 2512.15011 v3, preprint | Knowledge collapse in recursive training is worse in a model monoculture; optional, for the "same answers across models" framing. |
| padmakumar2024writing | arXiv 2309.05196, arXiv comment "ICLR 2024" (venue page not opened) | Writing with InstructGPT, unlike GPT-3, reduces content diversity across users, a human-side homogenisation effect of post-trained models. |
| anderson2024homogenization | Creativity and Cognition 2024, pp. 413 to 425; DOI 10.1145/3635636.3656204 (Crossref) | ChatGPT users produce less semantically distinct ideas than users of another tool; optional, for the human-side theme. |
| bommasani2022picking | arXiv 2211.13972, arXiv comment "Published at NeurIPS 2022" (venue page not opened) | Introduces outcome homogenisation under algorithmic monoculture; optional framing for "everyone's model omits the same entities". |
| sourati2025homogenizing | arXiv 2508.01491 v2, preprint (no venue stated) | Synthesis arguing LLMs standardise language and reasoning; optional, survey-style framing. |

Not proposed: Gerstgrasser et al. (arXiv 2404.01413, "Is Model Collapse Inevitable?"). The arXiv record states no venue, and none was verified. Add it only if the paper discusses mitigation of model collapse.

Recommended minimum for the introduction's theme paragraph: shumailov2024collapse, peterson2025collapse, guo2024curious, plus the existing jiang2025hivemind and kirk2024understanding. Wright et al. (EMNLP 2026) is the closest current knowledge-collapse measurement and is worth one sentence.

```bibtex
@article{shumailov2024collapse,
  title   = {{AI} models collapse when trained on recursively generated data},
  author  = {Shumailov, Ilia and Shumaylov, Zakhar and Zhao, Yiren and Papernot, Nicolas and Anderson, Ross and Gal, Yarin},
  journal = {Nature},
  volume  = {631},
  number  = {8022},
  pages   = {755--759},
  year    = {2024},
  doi     = {10.1038/s41586-024-07566-y}
}

@article{peterson2025collapse,
  title   = {{AI} and the problem of knowledge collapse},
  author  = {Peterson, Andrew J.},
  journal = {AI \& Society},
  volume  = {40},
  number  = {5},
  pages   = {3249--3269},
  year    = {2025},
  doi     = {10.1007/s00146-024-02173-x}
}

@inproceedings{dohmatob2024tails,
  title     = {A Tale of Tails: Model Collapse as a Change of Scaling Laws},
  author    = {Dohmatob, Elvis and Feng, Yunzhen and Yang, Pu and Charton, Francois and Kempe, Julia},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  pages     = {11165--11197},
  year      = {2024},
  url       = {https://proceedings.mlr.press/v235/dohmatob24b.html}
}

@inproceedings{guo2024curious,
  title     = {The Curious Decline of Linguistic Diversity: Training Language Models on Synthetic Text},
  author    = {Guo, Yanzhu and Shang, Guokan and Vazirgiannis, Michalis and Clavel, Chlo{\'e}},
  booktitle = {Findings of the Association for Computational Linguistics: NAACL 2024},
  year      = {2024},
  address   = {Mexico City, Mexico},
  publisher = {Association for Computational Linguistics},
  pages     = {3589--3604},
  doi       = {10.18653/v1/2024.findings-naacl.228},
  url       = {https://aclanthology.org/2024.findings-naacl.228/}
}

@inproceedings{alemohammad2024mad,
  title     = {Self-Consuming Generative Models Go {MAD}},
  author    = {Alemohammad, Sina and Casco-Rodriguez, Josue and Luzi, Lorenzo and Humayun, Ahmed Imtiaz and Babaei, Hossein and LeJeune, Daniel and Siahkoohi, Ali and Baraniuk, Richard},
  booktitle = {International Conference on Learning Representations (ICLR 2024)},
  year      = {2024}
}

@misc{wright2026epistemic,
  title         = {What and Whose Knowledge? Measuring Epistemic Diversity in Large Language Models},
  author        = {Wright, Dustin and Masud, Sarah and Moore, Jared and Yadav, Srishti and Antoniak, Maria and Christensen, Peter Ebert and Park, Chan Young and Augenstein, Isabelle},
  year          = {2025},
  eprint        = {2510.04226},
  archivePrefix = {arXiv},
  note          = {Accepted to EMNLP 2026 (arXiv comment)}
}

@misc{hodel2025ecosystem,
  title         = {Epistemic diversity across language models mitigates knowledge collapse},
  author        = {Hodel, Damian and West, Jevin D.},
  year          = {2025},
  eprint        = {2512.15011},
  archivePrefix = {arXiv}
}

@inproceedings{padmakumar2024writing,
  title     = {Does Writing with Language Models Reduce Content Diversity?},
  author    = {Padmakumar, Vishakh and He, He},
  booktitle = {International Conference on Learning Representations (ICLR 2024)},
  year      = {2024},
  note      = {Venue from the arXiv comment (arXiv 2309.05196)}
}

@inproceedings{anderson2024homogenization,
  title     = {Homogenization Effects of Large Language Models on Human Creative Ideation},
  author    = {Anderson, Barrett R. and Shah, Jash Hemant and Kreminski, Max},
  booktitle = {Proceedings of the 16th Conference on Creativity and Cognition},
  pages     = {413--425},
  year      = {2024},
  doi       = {10.1145/3635636.3656204}
}

@inproceedings{bommasani2022picking,
  title     = {Picking on the Same Person: Does Algorithmic Monoculture lead to Outcome Homogenization?},
  author    = {Bommasani, Rishi and Creel, Kathleen A. and Kumar, Ananya and Jurafsky, Dan and Liang, Percy},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2022)},
  year      = {2022},
  note      = {Venue from the arXiv comment (arXiv 2211.13972)}
}

@misc{sourati2025homogenizing,
  title         = {The Homogenizing Effect of Large Language Models on Human Expression and Thought},
  author        = {Sourati, Zhivar and Ziabari, Alireza S. and Dehghani, Morteza},
  year          = {2025},
  eprint        = {2508.01491},
  archivePrefix = {arXiv}
}
```

Notes on the entries:
- "Proceedings of the 16th Conference on Creativity and Cognition" for Anderson et al.: Crossref gives the container as "Creativity and Cognition" and the pages. The edition number comes from C&C 2024 being the 16th conference; check it on the ACM DL page before submission, or write booktitle = {Creativity and Cognition (C\&C 2024)} instead.
- The proceedings.neurips.cc and proceedings.iclr.cc hosts refused connections from this machine. ICLR and NeurIPS venues were confirmed on iclr.cc and neurips.cc instead; volume numbers and pages for those are left out on purpose.

## 2. DPO

Verified on neurips.cc/virtual/2023/poster/72164 (title, NeurIPS 2023, oral and poster, "Outstanding Paper Runner-up") and arXiv 2305.18290. The two sources order the authors differently:
- arXiv: Rafailov, Sharma, Mitchell, **Ermon, Manning**, Finn.
- NeurIPS page: Rafailov, Sharma, Mitchell, **Manning, Ermon**, Finn.

The entry uses the proceedings order. proceedings.neurips.cc could not be reached, so the volume and pages are left out.

```bibtex
@inproceedings{rafailov2023dpo,
  title     = {Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author    = {Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Manning, Christopher D. and Ermon, Stefano and Finn, Chelsea},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2023)},
  year      = {2023},
  eprint    = {2305.18290},
  archivePrefix = {arXiv}
}
```

Use: cite where the paper names the OLMo-2 preference stage ("after preference tuning (DPO)") and in the sentence "the KL-regularised preference optimum keeps the support of the reference model". That sentence is DPO's closed-form optimum, so the citation is needed there.

## 3. Licences (official model cards and pages)

The licence field was read from each model card through the Hugging Face API (https://huggingface.co/api/models/<id>). In every case the card's commit is the snapshot our runs used.

| artefact | snapshot used | licence | official source |
|---|---|---|---|
| Qwen/Qwen2.5-7B | d149729 | Apache 2.0 | https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/LICENSE |
| Qwen/Qwen2.5-7B-Instruct | a09a354 | Apache 2.0 | https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE |
| Qwen/Qwen2.5-0.5B-Instruct (scale ladder) | 7ae5576 | Apache 2.0 | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE |
| Qwen/Qwen2.5-1.5B-Instruct (scale ladder) | 989aa79 | Apache 2.0 | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE |
| **Qwen/Qwen2.5-3B-Instruct (scale ladder)** | aa8e725 | **Qwen Research License Agreement** (card: license "other", name "qwen-research"; file header "Qwen RESEARCH LICENSE AGREEMENT Release Date: September 19, 2024") | https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE |
| allenai/OLMo-2-1124-7B | 7df9a82 | Apache 2.0 (card: "The code and model are released under Apache 2.0.") | https://huggingface.co/allenai/OLMo-2-1124-7B |
| allenai/OLMo-2-1124-7B-SFT | 1de02c0 | Apache 2.0 | https://huggingface.co/allenai/OLMo-2-1124-7B-SFT |
| allenai/OLMo-2-1124-7B-DPO | e34ea60 | Apache 2.0 | https://huggingface.co/allenai/OLMo-2-1124-7B-DPO |
| allenai/OLMo-2-1124-7B-Instruct | 470b1fb | Apache 2.0 | https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct |
| meta-llama/Llama-3.1-8B (pending run) | d04e592 | Llama 3.1 Community License Agreement (card: license "llama3.1"; gated, manual approval) | https://huggingface.co/meta-llama/Llama-3.1-8B ; licence text https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE |
| meta-llama/Llama-3.1-8B-Instruct (pending run) | 0e9e39f | Llama 3.1 Community License Agreement | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct ; same licence text |
| Wikidata (TailCat labels, aliases, sitelink counts) | query snapshots in data/tailcat/*/raw | CC0 1.0 for structured data: "All structured data in the main, Property, Lexeme, and EntitySchema namespaces is released into the public domain under Creative Commons Zero"; text in other namespaces is CC BY-SA 4.0 | https://www.wikidata.org/wiki/Wikidata:Licensing |

Points for the paper and the ARR checklist:

1. **Qwen2.5-3B-Instruct is not Apache 2.0.** It is used only in the exploratory scale ladder. The Qwen Research License covers research use, which ours is; say so in the licence statement.
2. **Llama 3.1 terms that matter.** Section 1.b.i asks anyone who "distribute[s] or make[s] available the Llama Materials (or any derivative works thereof)" to "prominently display 'Built with Llama'". Section 1.b.iii requires a notice in redistributed copies. Section 1.b.iv requires use to "adhere to the Acceptable Use Policy" (https://llama.meta.com/llama3_1/use-policy).
   - We do not redistribute weights.
   - Our reading is that publishing measurements and sampled answers is research use that triggers neither 1.b.i nor 1.b.iii. The human should confirm this reading before release.
   - Either way, state compliance with the Acceptable Use Policy and name the licence in the paper.
3. **TailCat release** uses only Wikidata structured data, so it can be released under CC0. The prompts and sampled answers we release should follow each model's terms (Apache 2.0 for Qwen 7B and OLMo; Llama 3.1 terms for Llama outputs).

Proposed licence sentence for the paper (ethics or appendix):

> Qwen2.5-7B, Qwen2.5-7B-Instruct, Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct and all OLMo-2-1124-7B checkpoints are released under Apache 2.0; Qwen2.5-3B-Instruct under the Qwen Research License; Llama-3.1-8B and Llama-3.1-8B-Instruct under the Llama 3.1 Community License, whose Acceptable Use Policy we follow. We use all models for research only and do not redistribute weights. TailCat is built from Wikidata structured data, which is released under CC0, and we release it under CC0.
