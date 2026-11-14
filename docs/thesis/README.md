# Thesis documents

**Authors:** Lê Minh Nhân and Bùi Quang Minh.
**Supervisor:** ThS. Trương Ngọc Hà.
**Language:** Vietnamese. **Academic period:** July 2026.

| Document | Download | Public copy |
| --- | --- | --- |
| Thesis report | [PDF](https://github.com/MinhNhan0508/channel-prediction/releases/download/v1.0.2/dast-thesis-report.vi.pdf) | 82 pages |
| Defense presentation | [PDF](https://github.com/MinhNhan0508/channel-prediction/releases/download/v1.0.2/dast-thesis-slides.vi.pdf) | 24 slides |

The PDFs are Release assets rather than files in Git history. Source packages
include these notes and [checksums](documents.json), not the PDF binaries.

## Public-copy preparation

- The original report has 87 pages. Original PDF pages 3–7 contain administrative
  revision/review forms and signatures; those five pages are excluded.
- Student numbers are removed from the report's two covers and assignment page,
  and from the presentation's title slide. Names and academic credits remain.
- No technical text, equations, plots, result tables or citations have been rewritten.
  Original chapter/table/page labels remain, so a printed page number may differ
  from its position in the public PDF.
- The original files remain unchanged. The public copies are not certified
  administrative records or verbatim copies of every page of the submission.

## Thesis-era results and the current benchmark

The report and slides document the academic work; they are not the source of the
new benchmark numbers in the repository README. In the maintained experiment,
beam selection and observation-noise calibration use history only, data and model
identities are recorded, and models are retrained and evaluated under that policy.
The original report's results therefore need not match the current tables.

Use the [current benchmark](../benchmark-results/README.md) and
[methodology](../methodology.md) for reproducible measurements and assumptions.
Historical BER, sum-rate and latency plots are retained as part of the thesis,
not presented as newly reproduced results of the maintained benchmark.

The implementation's classical prior is coefficient-wise AR(4), not a full
angle-delay-domain PAD reproduction. Component ablations in the maintained
benchmark do not establish that FiLM or an adaptive gate always improves results.

## Attribution

The PDFs retain the two authors' names, the supervisor's name and the original
citations. They do not change the thesis into a solo academic submission.
The repository's MIT license does not relicense university marks or third-party
figures and other cited material in these documents.
