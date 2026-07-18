import httpx
import pytest

from crawler.github_client import GitHubClient, GitHubRequestError


def test_follows_link_header_until_next_is_absent():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if "page=2" in str(request.url):
            return httpx.Response(200, json=[{"id": 2}])
        return httpx.Response(
            200,
            json=[{"id": 1}],
            headers={"Link": '<https://api.github.test/items?page=2>; rel="next"'},
        )

    client = GitHubClient(
        "token", base_url="https://api.github.test", transport=httpx.MockTransport(handler)
    )
    try:
        pages = list(client.paginate("/items", {"per_page": 100}))
    finally:
        client.close()
    assert [page.items[0]["id"] for page in pages] == [1, 2]
    assert len(calls) == 2


def test_page_limit_stops_before_following_next_link():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        page = len(calls)
        return httpx.Response(
            200,
            json=[{"id": page}],
            headers={
                "Link": f'<https://api.github.test/items?page={page + 1}>; rel="next"'
            },
        )

    client = GitHubClient(
        "token", base_url="https://api.github.test", transport=httpx.MockTransport(handler)
    )
    try:
        pages = list(client.paginate("/items", {}, max_pages=2))
    finally:
        client.close()
    assert [page.items[0]["id"] for page in pages] == [1, 2]
    assert len(calls) == 2


def test_retries_recoverable_server_error():
    attempts = 0
    sleeps = []

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"message": "try later"})
        return httpx.Response(200, json=[])

    client = GitHubClient(
        "token",
        base_url="https://api.github.test",
        sleep=sleeps.append,
        random_uniform=lambda _a, _b: 0,
        transport=httpx.MockTransport(handler),
    )
    try:
        page = next(client.paginate("/items", {"per_page": 100}))
    finally:
        client.close()
    assert page.retries == 1
    assert attempts == 2
    assert sleeps == [1]


def test_does_not_retry_authentication_error():
    def handler(request):
        return httpx.Response(401, json={"message": "Bad credentials"})

    client = GitHubClient("bad", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(GitHubRequestError) as error:
            next(client.paginate("/items", {}))
    finally:
        client.close()
    assert error.value.status_code == 401
