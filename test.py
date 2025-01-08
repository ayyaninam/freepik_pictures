import httpx
import asyncio

async def test_freepik():
    headers = {
        'authority': 'www.freepik.com',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-language': 'en-US,en;q=0.9,el;q=0.8,ur;q=0.7,sl;q=0.6',
        'cache-control': 'no-cache',
        'dnt': '1',
        'pragma': 'no-cache',
        'sec-ch-ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'sec-fetch-dest': 'document',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-site': 'none',
        'sec-fetch-user': '?1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
    }

    cookies = {
        '_cs_ex': '1709818470',
        '_cs_c': '0',
        'usprivacy': '1---',
        'OptanonAlertBoxClosed': '2024-09-21T10:39:20.541Z',
        'OneTrustWPCCPAGoogleOptOut': 'false',
        '_au_1d': 'AU1D-0100-001709006278-QV0BJASJ-L0AE',
        '_hjSessionUser_1331604': 'eyJpZCI6IjJhZjliM2IwLWQ5ZDMtNWQwOC1iYzVjLTU2ZWM2ZWQyNzY3ZCIsImNyZWF0ZWQiOjE3MjY5MTUxNjExMTUsImV4aXN0aW5nIjp0cnVlfQ==',
        'cto_optout': '1',
        'GRID': '71985952',
        'premiumGen': 'B',
        'EXPCH': 'true',
        'new_regular_detail_test': 'A',
        'TUNES_IN_VIDEO': '1',
        'premiumQueue': 'A',
        'skip_expch': 'true'
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                'https://www.freepik.com/api/regular/search',
                params={
                    'filters[ai-generated][only]': '1',
                    'filters[content_type]': 'photo',
                    'locale': 'en',
                    'term': 'SWOT Analysis Best Practices'
                },
                headers=headers,
                cookies=cookies,
                follow_redirects=True
            )
            
            print(f"Status: {response.status_code}")
            print(f"Headers: {dict(response.headers)}")
            print(f"Response: {response.text[:500]}")

        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_freepik())