---
layout: default
title: Blog
permalink: /blog/
---

<ul class="blog-list">
{% assign posts = site.pages | where: "layout", "post" | sort: "date" | reverse %}
{% for post in posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    {% if post.description %}
      <span class="description"> — {{ post.description }}</span>
    {% endif %}
  </li>
{% endfor %}
</ul>
