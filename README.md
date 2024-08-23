# Looking Right is Sometimes Right: Investigating the Capabilities of Decoder-only LLMs for Sequence Labeling

Code for the paper [Looking Right is Sometimes Right: Investigating the Capabilities of Decoder-only LLMs for Sequence Labeling](https://arxiv.org/abs/2401.14556) accepted at ACL 2024 Findings.

## Main idea
Layer-wise causal mask (CM) removal in decoder-only LLMs can drastically improve their performance on sequence labeling (SL) tasks. This approach yields performance gains competitive with state-of-the-art SL models, matching or outperforming the results of CM removal from all blocks. Our findings hold for diverse SL tasks, demonstrating that open LLMs with layer-dependent CM removal outperform strong MLM-based encoders and even instruction-tuned LLMs. 

## Citing
```
@inproceedings{dukic-snajder-2024-looking,
    title = "Looking Right is Sometimes Right: Investigating the Capabilities of Decoder-only {LLM}s for Sequence Labeling",
    author = "Duki{\'c}, David  and
      Snajder, Jan",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.843",
    pages = "14168--14181",
}
```
