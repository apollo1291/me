# Ellington Hemphill

Personal site built with Jekyll, hosted on GitHub Pages.

**Live site:** [ellingtonhemphill.github.io/me](https://ellingtonhemphill.github.io/me/)

## GitHub Pages setup

1. Create the repo `ellingtonhemphill/me` on GitHub (if you haven't already).
2. Point your remote:
   ```bash
   git remote set-url origin https://github.com/ellingtonhemphill/me.git
   ```
3. Push to `main`.
4. In the repo: **Settings → Pages → Build and deployment → Source: Deploy from branch `main`, folder `/ (root)`**.
5. Wait a minute or two, then visit [ellingtonhemphill.github.io/me](https://ellingtonhemphill.github.io/me/).

## Adding a blog post

1. Create a folder: `blog/<slug>/`
2. Add `blog.md` with front matter:

   ```yaml
   ---
   title: my-post
   description: a short one-line summary
   date: 2026-06-22
   ---
   ```

3. Add images in the same folder and reference them by filename:

   ```markdown
   ![caption](my-chart.png)
   ```

Or with raw HTML (do not indent the tags — Kramdown treats indented HTML as code):

<p align="center">
<img src="my-photo.png" alt="description" width="300">
</p>

4. Push — the post appears on `/blog` automatically.

## Local preview

`github-pages` does not support Ruby 4 yet. Use **Ruby 3.3**:

```bash
brew install ruby@3.3
export PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH"
ruby --version   # should show 3.3.x
```

Then from the project folder:

```bash
cd /Users/ellingtonhemphill/personal_site
bundle install
bundle exec jekyll serve
```

Open [http://localhost:4000/me/](http://localhost:4000/me/).

To make Ruby 3.3 permanent, add the `export PATH=...` line to `~/.zshrc`.

**If you already ran `bundle install` with Ruby 4**, delete `Gemfile.lock` and re-run `bundle install` after switching to 3.3.
