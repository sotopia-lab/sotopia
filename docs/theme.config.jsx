
export default {
    logo:
    <div className="flex items-center">
      <svg width="24" height="24" viewBox="0 0 300 300">
        <path d="M150,108.33c0,46.03,55.96,83.33,125,83.33V25c-69.04,0-125,37.31-125,83.33Z" fill="currentColor" strokeWidth="0"/>
        <path d="M150,191.67c0-46.03-55.96-83.33-125-83.33v166.67c69.04,0,125-37.31,125-83.33Z" fill="currentColor" strokeWidth="0"/>
        <path d="M150,25h-65c-33.14,0-60,26.86-60,60v23.33c69.04,0,125-37.31,125-83.33Z" fill="currentColor" strokeWidth="0"/>
        <path d="M150,275h65c33.14,0,60-26.86,60-60v-23.33c-69.04,0-125,37.31-125,83.33Z" fill="currentColor" strokeWidth="0"/>
      </svg>
    <span className="text-2xl font-display">Sotopia</span>
    </div>,
    logoLink: "https://docs.sotopia.world",
    project: {
      link: 'https://github.com/sotopia-lab/sotopia',
    },
    toc: {
      backToTop: true,
    },
    sidebar: {
      toggleButton: true,
    },
    search: {
      placeholder: 'Search contents',
    },
    feedback: {
        content: null,
    },
    head: (
      <>
        <link rel="icon" href="/favicon.ico" type="image/ico" />
        <link rel="icon" href="/favicon.svg" type="image/svg" />
      </>
    ),
    footer: {
      text: (
        <span>
          MIT {new Date().getFullYear()} ©{' '}
          <a href="https://sotopia.world" target="_blank">
            Sotopia Lab
          </a>
          .
        </span>
      )
  },
    useNextSeoProps() {
      return {
        titleTemplate: '%s – sotopia',
        description: '',
        openGraph: {
            type: 'website',
            images: [
              {
                url: 'https://github.com/sotopia-lab/sotopia/raw/main/figs/title.png',
              }
            ],
            locale: 'en_US',
            url: 'https://sotopia.world',
            siteName: 'Sotopia',
            title: 'Sotopia',
            description: 'Sotopia: an Open-ended Social Learning Environment',
        },
        twitter: {
            cardType: 'summary_large_image',
            title: 'Sotopia: an Open-ended Social Learning Environment',
            image: 'https://github.com/sotopia-lab/sotopia/raw/main/figs/title.png',
        },
      }

  },
    // ... other theme options
  }
