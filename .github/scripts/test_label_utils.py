from typing import Any

from unittest import TestCase, mock, main
from label_utils import (
    get_last_page_num_from_header,
    gh_get_labels,
)


class TestLabelUtils(TestCase):
    MOCK_HEADER_LINKS_TO_PAGE_NUMS = {
        1: {"link": "<https://api.github.com/dummy/labels?per_page=10&page=1>; rel='last'"},
        2: {"link": "<https://api.github.com/dummy/labels?per_page=1&page=2>;"},
        3: {"link": "<https://api.github.com/dummy/labels?per_page=1&page=2&page=3>;"},
    }

    def test_get_last_page_num_from_header(self) -> None:
        for expected_page_num, mock_header in self.MOCK_HEADER_LINKS_TO_PAGE_NUMS.items():
            self.assertEqual(get_last_page_num_from_header(mock_header), expected_page_num)

    MOCK_LABEL_INFO = '[{"name": "foo"}]'

    @mock.patch("label_utils.get_last_page_num_from_header", return_value=3)
    @mock.patch("label_utils.request_for_labels", return_value=(None, MOCK_LABEL_INFO))
    def test_gh_get_labels(
        self,
        mock_request_for_labels: Any,
        mock_get_last_page_num_from_header: Any,
    ) -> None:
        res = gh_get_labels("mock_org", "mock_repo")
        mock_get_last_page_num_from_header.assert_called_once()
        self.assertEqual(res, ["foo"] * 3)

    @mock.patch("label_utils.get_last_page_num_from_header", return_value=0)
    @mock.patch("label_utils.request_for_labels", return_value=(None, MOCK_LABEL_INFO))
    def test_gh_get_labels_raises_with_no_pages(
        self,
        mock_request_for_labels: Any,
        get_last_page_num_from_header: Any,
    ) -> None:
        with self.assertRaises(AssertionError) as err:
            gh_get_labels("foo", "bar")
        self.assertIn("number of pages of labels", str(err.exception))


if __name__ == "__main__":
    main()
