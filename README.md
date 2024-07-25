# Looking Right is Sometimes Right: Investigating the Capabilities of Decoder-only LLMs for Sequence Labeling

Code for the paper [Looking Right is Sometimes Right: Investigating the Capabilities of Decoder-only LLMs for Sequence Labeling](https://arxiv.org/abs/2401.14556) accepted at ACL 2024 Findings.

## Main idea
Layer-wise causal mask removal in decoder-only LLMs can drastically improve their performance on sequence labeling (SL) tasks. This approach yields performance gains competitive with state-of-the-art SL models, matching or outperforming the results of CM removal from all blocks. Our findings hold for diverse SL tasks, demonstrating that open LLMs with layer-dependent CM removal outperform strong MLM-based encoders and even instruction-tuned LLMs. 

## Citing
```
@article{dukic2024looking,
  title={Looking right is sometimes right: Investigating the capabilities of decoder-only llms for sequence labeling},
  author={Dukic, David and {\v{S}}najder, Jan},
  journal={arXiv preprint arXiv:2401.14556},
  year={2024}
}
```
