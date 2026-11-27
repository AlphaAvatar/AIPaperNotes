import type { NextConfig } from 'next'
import nextra from 'nextra'

const withNextra = nextra({
  latex: true
})

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Build the whole site into `out/` as plain static files (no ISR / Functions)
  output: 'export',
  // Required by `output: 'export'`; content images are remote URLs rendered as <img>
  images: { unoptimized: true }
}

export default withNextra(nextConfig)
