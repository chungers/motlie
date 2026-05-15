# npm Auth

Use this reference when publishing Motlie packages to npm under `@motlie`.

Parameterize examples by the current per-binary manifest, not by workspace-level binary fields:

```text
BINARY_MANIFEST=releases/<bin>.toml
[identity].binary=<installed command name>
[identity].version=<release version>
[npm].bin_command=<installed command name>
[npm].bin_path=<package binary path>
[[target]].npm_package=<platform package name, e.g. @motlie/<bin>-linux-x64-musl>
PACKAGE_DIR=<generated package directory for [[target]].npm_package>
```

Read package names and binary paths from `BINARY_MANIFEST` when explicit fields are present. For native packages, `runner = "native-binary"` and `node_launcher = false` means the npm package must expose the binary directly and must not create `<bin>.js` or `<bin>.sh` as the runtime entrypoint.

Native package directory shape:

```text
package.json
README.md
LICENSE
bin/<bin>
```

Minimal package contract:

```json
{
  "name": "<[[target]].npm_package>",
  "version": "<[identity].version>",
  "bin": {
    "<[npm].bin_command>": "<[npm].bin_path>"
  },
  "files": ["bin", "README.md", "LICENSE"]
}
```

Package staging:

1. Copy the final signed/release binary to the manifest `npm_bin_path`.
2. Generate `package.json` from manifest values; do not derive package names when `npm_package` is explicit.
3. Run `npm pack --dry-run`.
4. Run `npm pack`, install the generated `.tgz` on a matching host, and execute `<[npm].bin_command> --version`.
5. Record dry-run output, `.tgz` checksum, install-test output, package name, version, and source tag in the target-specific `npm-published` gate before publishing.

Preferred path:

- use npm trusted publishing from GitHub Actions
- configure package publishing in npm for the release workflow
- require `permissions.id-token: write`
- keep npm provenance enabled; trusted publishing can attach registry provenance to the published package without a long-lived token
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
- release-branch status updates

Token is needed only for:

- `npm publish --access public`

Minimal publish step with token fallback:

```yaml
- uses: actions/setup-node@v4
  with:
    node-version: 24
    registry-url: https://registry.npmjs.org

- run: npm publish --access public
  working-directory: ${PACKAGE_DIR}
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
    working-directory: ${PACKAGE_DIR}
```

For each `npm-published` gate, record the package name, version, registry URL, provenance mode, and install-test evidence in the manifest. Do not print token values, commit `.npmrc` with credentials, or reuse personal all-access tokens.
