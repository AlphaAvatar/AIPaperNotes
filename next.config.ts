import nextra from 'nextra'

const withNextra = nextra({
  latex: true
})

const nextConfig = {
  reactStrictMode: true
}

export default withNextra(nextConfig)
