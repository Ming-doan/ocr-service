# ViettelPost1

## How to run

1. Run vllm service

```sh
docker compose -f docker-compose.vllm.yml up -d
```

2. Run API service

```sh
docker compose docker-compose.yml up -d
```

3. Run automation scripts

```sh
python scripts/export_pdf_public_test.py --in-dir ./pdfs --out-dir ./output
```

4. Run final format to final result

```sh
python scripts/format_to_result.py --res-dir ./output --out-dir ./OCR.zip
```
