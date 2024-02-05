## P9 Presentation

Install [Quarto](https://quarto.org/) to run the presentation.
You can also grab the [VSCode extension](https://marketplace.visualstudio.com/items?itemName=quarto.quarto).

How to run:

```bash
quarto preview
```

### Editing workflow

Since [Quarto currently doesn't support hot-reload of included files](https://github.com/quarto-dev/quarto-cli/issues/2795), please work on your slides in the main document during development, and then place them in the designated file in `sections` when you're done.
This way, we can keep the main document clean and easy to read.
