# npm Auth

Use this reference when publishing Motlie packages to npm under `@motlie`.

Preferred path:

- use npm trusted publishing from GitHub Actions
- configure package publishing in npm for the release workflow
- require `permissions.id-token: write`
- do not store a long-lived npm token

Fallback bootstrap path:

1. Create a granular npm token scoped to the `@motlie` org/packages.
2. Store it in the source repo as a GitHub Actions secret named `NPM_TOKEN`.
3. Use the secret only in the `npm publish` step.
4. Pass it as `NODE_AUTH_TOKEN`.
5. Revoke/remove the token after trusted publishing works.

Token is not needed for:

- cross-compiling binaries
- creating tar archives
- uploading GitHub Release assets
- running `npm pack --dry-run`
- local package install verification from a `.tgz`
- Homebrew formula or bottle work

Token is needed only for:

- `npm publish --access public`

Minimal publish step with token fallback:

```yaml
- uses: actions/setup-node@v4
  with:
    node-version: 24
    registry-url: https://registry.npmjs.org

- run: npm publish --access public
  working-directory: dist/npm/@motlie/mmux-linux-x64-gnu
  env:
    NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

Trusted publishing shape:

```yaml
permissions:
  contents: read
  id-token: write

steps:
  - uses: actions/setup-node@v4
    with:
      node-version: 24
      registry-url: https://registry.npmjs.org
  - run: npm publish --access public
    working-directory: dist/npm/@motlie/mmux-linux-x64-gnu
```

Do not print token values, commit `.npmrc` with credentials, or reuse personal all-access tokens.
