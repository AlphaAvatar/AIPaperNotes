import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import { Head } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'
import 'katex/dist/katex.min.css'

export const metadata = {
  title: 'AI Paper Notes',
  description:
    'Paper reading notes for LLM, multimodal, agent, training and inference topics.'
}

const navbar = (
  <Navbar
    logo={<b>AI Paper Notes</b>}
    projectLink="https://github.com/AlphaAvatar/AIPaperNotes"
  />
)

const footer = <Footer>© 2026 AI Paper Notes</Footer>

export default async function RootLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" dir="ltr" suppressHydrationWarning>
      <Head />
      <body>
        <Layout
          navbar={navbar}
          footer={footer}
          pageMap={await getPageMap()}
          docsRepositoryBase="https://github.com/AlphaAvatar/AIPaperNotes/tree/main"
          sidebar={{ defaultMenuCollapseLevel: 1 }}
          editLink="Edit this page on GitHub"
          feedback={{ content: 'Give feedback' }}
          navigation={false}
        >
          {children}
        </Layout>
      </body>
    </html>
  )
}
