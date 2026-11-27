// Local preview of the static export in `out/`, routed the way vercel.json
// serves it: `X.html` is served at `X` and `/X.html` redirects there
// (cleanUrls), a trailing slash redirects with 308 (trailingSlash: false),
// unknown paths get 404.html with status 404, and directories are never listed.
import { createReadStream, statSync } from 'node:fs'
import { createServer } from 'node:http'
import { extname, join, resolve, sep } from 'node:path'

const root = resolve('out')
const port = Number(process.env.PORT) || 3000

const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.ttf': 'font/ttf',
  '.wasm': 'application/wasm'
}

const isFile = path => {
  try {
    return statSync(path).isFile()
  } catch {
    return false
  }
}

const notFoundPage = join(root, '404.html')
if (!isFile(notFoundPage)) {
  console.error('out/404.html not found. Run `pnpm build` first.')
  process.exit(1)
}

function findFile(pathname) {
  if (pathname === '/') return join(root, 'index.html')
  const target = join(root, pathname)
  if (!target.startsWith(root + sep)) return null
  if (isFile(target)) return target
  return isFile(target + '.html') ? target + '.html' : null
}

function send(req, res, status, file) {
  const headers = {
    'Content-Type': contentTypes[extname(file)] || 'application/octet-stream',
    'Content-Length': statSync(file).size,
    'Cache-Control': req.url.startsWith('/_next/static/')
      ? 'public, max-age=31536000, immutable'
      : 'public, max-age=0, must-revalidate'
  }
  res.writeHead(status, headers)
  if (req.method === 'HEAD') res.end()
  else createReadStream(file).pipe(res)
}

function redirect(res, location) {
  res.writeHead(308, { Location: location }).end()
}

createServer((req, res) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.writeHead(405, { Allow: 'GET, HEAD' }).end()
    return
  }
  const url = new URL(req.url, 'http://localhost')
  const cleanPath = url.pathname.replace(/\.html\/?$/, '')
  if (cleanPath !== url.pathname) {
    redirect(res, (cleanPath || '/') + url.search)
    return
  }
  if (url.pathname.length > 1 && url.pathname.endsWith('/')) {
    redirect(res, (url.pathname.replace(/\/+$/, '') || '/') + url.search)
    return
  }
  let pathname
  try {
    pathname = decodeURIComponent(url.pathname)
  } catch {
    res.writeHead(400).end()
    return
  }
  const file = findFile(pathname)
  if (file) send(req, res, 200, file)
  else send(req, res, 404, notFoundPage)
}).listen(port, () => {
  console.log(`Serving out/ at http://localhost:${port}`)
})
