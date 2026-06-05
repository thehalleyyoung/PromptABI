# Training-pipeline quickstart

This quickstart verifies a tiny chat SFT dataset without GPUs, weights, or
network access. The manifest records the structural facts emitted by a data
builder: rendered roles, token spans, source attribution hashes, packing
boundaries, loss masks, tokenizer/template stage pins, and witness-redaction
policy.

```bash
promptabi verify --config examples/end-to-end/training-quickstart/buggy.promptabi.json
promptabi verify --config examples/end-to-end/training-quickstart/fixed.promptabi.json
```

The buggy manifest lets user text overlap a supervised assistant target and
crosses a preserved packed-example boundary. The fixed manifest proves the same
toy dataset's assistant target stays inside the rendered assistant region, is
loss-masked, survives packing, uses one tokenizer/template contract from data
prep through serving, and stores only hashed or structural evidence.
