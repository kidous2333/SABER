# SABER Project Website

Static single-page website for the SABER project (UDMT-style minimal academic
layout). No build step, no external dependencies.

Live URL: **https://sunlabsaber.netlify.app** (Netlify)

## Structure

```
SABER_site/
├── index.html            # sections: Title / Figure 1 / Introduction / Method /
│                         #   GUI / Results / Availability
├── css/style.css         # all styling (UDMT-like: centered titles + <hr>)
├── assets/
│   ├── figures/          # main figures (rendered from assets/Figure_09102026/*.pdf)
│   ├── img/              # icon.png, favicon.png, apple-touch-icon.png, architecture.png
│   └── videos/           # demo videos go here (not used yet)
└── README.md
```

## Local preview

```bash
cd SABER_site
python -m http.server 8000
# open http://localhost:8000
```

## Updating the live site

Re-upload the whole `SABER_site` folder on Netlify (site Deploys page → drag &
drop, or https://app.netlify.com/drop). The URL stays the same.

## Remaining TODOs (search `TODO` in index.html)

1. Hero author list / affiliations (paper v7 docx has no plain-text author
   block; fill from the submission system).
2. Paper link (add once published).
3. GUI screenshot (drop PNG into `assets/img/`, replace the placeholder div).
4. Dataset / pretrained-model download links (once hosted).

## Content sources

- Title / abstract / method text: `论文初稿/SABER NM 7th version 08252026.docx`
- Main figures: `assets/Figure_09102026/Figure1..5.pdf` (rendered at 2x → `assets/figures/Figure1..5.png`)
- GitHub: https://github.com/kidous2333/SABER
