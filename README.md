# Ellington Hemphill

Personal site: [ellingtonhemphill.com](https://ellingtonhemphill.com)

## Local preview

`github-pages` does not work with Ruby 4. Use **Ruby 3.3**:

```bash
export PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH"
ruby --version   # should show 3.3.x
cd /Users/ellingtonhemphill/personal_site
bundle install
bundle exec jekyll serve
```

Open [http://localhost:4000/](http://localhost:4000/).

Add the `export PATH=...` line to `~/.zshrc` so new terminals use Ruby 3.3 by default.
