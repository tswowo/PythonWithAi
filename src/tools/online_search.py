import os
import requests

DS_Search_URL = "https://api.deepseek.com/v1/tools/search"
DS_FETCH_URL = "https://api.deepseek.com/v1/tools/fetch"


def search_online(query: str, count: int = 10, ori_question: str = "", localization: str = "",
                  need_content: bool = False):
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    response = requests.post(
        DS_Search_URL,
        headers={
            "Authorization": f"Bearer {deepseek_api_key}",
        },
        json={
            "query": query,
            "count": count,
            "ori_question": ori_question,
            "localization": localization,
            "need_content": need_content,
        },
    )
    response.raise_for_status()

    result = response.json()
    print(result)
    return result


def fetch_online(url: str):
    deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
    response = requests.post(
        DS_FETCH_URL,
        headers={
            "Authorization": f"Bearer {deepseek_api_key}",
        },
        json={"url": url},
    )
    response.raise_for_status()

    page = response.json()
    print(page)
    return page
