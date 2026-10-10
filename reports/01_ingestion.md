# 01 — Data Ingestion Report

## Data Source

National Highway Traffic Safety Administration (NHTSA)

Website: https://www.nhtsa.gov/

## Initial Crawling Attempt

Date: October 10, 2026

The team implemented a Python crawler using Requests
and BeautifulSoup.

The crawler includes:
- robots.txt verification
- Request rate limiting
- Descriptive User-Agent
- Raw document storage
- SHA-256 content hashing
- Duplicate detection

## Access Restriction

During initial testing, NHTSA returned HTTP 403 Forbidden
for both the homepage and robots.txt.

The crawler stopped because crawling permissions could
not be verified.

## Resolution Status

The team is investigating official NHTSA APIs and
other authorized public data sources.

No restricted pages were collected.