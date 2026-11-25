---
name: summarize-hacker-news-front-page-titles
description: >-
  Fetch the first page of Hacker News and summarize what its front-page
  titles are about. Use when the user asks to summarize the Hacker News
  front page, what is trending on Hacker News, or the current front-page
  titles.
---

# summarize-hacker-news-front-page-titles

## Instructions

1. Use the fetch_page tool to fetch https://news.ycombinator.com/.
2. Extract the titles of the first 30 stories on the page, in rank order,
   together with each story's point count and comment count. Ignore posters.
3. Summarize the titles themselves — there is no need to open or read the
   linked articles:
   - Group the titles into a few themes (e.g. AI, security, programming
     languages, hardware, startups) and describe each theme in 1-3 sentences.
   - Highlight any title that stands out, with a brief reason why; use the
     point and comment counts to flag the posts with the most engagement
     (mention their points and comments when you do).
4. If the fetch fails, tell the user that the page could not be retrieved.
   Do not fall back on memory of what is typically on the front page.
