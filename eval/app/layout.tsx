import '@mantine/core/styles.css';
import './globals.css';

import React from 'react';
import { ColorSchemeScript, mantineHtmlProps, MantineProvider } from '@mantine/core';
import { theme } from '../theme';

const BASE_URL = 'https://eval.codigollm.es';

export const metadata = {
  metadataBase: new URL(BASE_URL),
  title: {
    default: 'eval · Codigo LLM — Verificación de noticias IA',
    template: '%s · eval · Codigo LLM',
  },
  description:
    'Verificación sistemática de noticias sobre Inteligencia Artificial Generativa. Cada claim evaluado contra fuentes primarias. Un proyecto de Codigo LLM.',
  alternates: {
    canonical: BASE_URL,
    languages: { 'es-ES': BASE_URL },
  },
  openGraph: {
    title: 'eval · Codigo LLM — Verificación de noticias IA',
    description:
      'Verificación sistemática de noticias sobre IA. Cada claim evaluado contra fuentes primarias.',
    siteName: 'eval · Codigo LLM',
    type: 'website',
    locale: 'es_ES',
  },
  twitter: {
    card: 'summary_large_image',
    creator: '@jddam',
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({ children }: { children: any }) {
  const websiteJsonLd = {
    '@context': 'https://schema.org',
    '@type': 'WebSite',
    name: 'eval · Codigo LLM',
    url: BASE_URL,
    description: 'Verificación sistemática de noticias sobre Inteligencia Artificial Generativa.',
    inLanguage: 'es',
    isPartOf: {
      '@type': 'WebSite',
      name: 'Codigo LLM',
      url: 'https://www.codigollm.es',
    },
    author: {
      '@type': 'Person',
      name: 'Albert Gil López',
      url: 'https://albertgilopez.com',
      jobTitle: 'CTO',
      worksFor: { '@type': 'Organization', name: 'M.IA', url: 'https://predicta.es' },
      sameAs: [
        'https://linkedin.com/in/albertgilopez',
        'https://github.com/albertgilopez',
        'https://x.com/jddam',
      ],
    },
  };

  return (
    <html lang="es" {...mantineHtmlProps}>
      <head>
        <ColorSchemeScript defaultColorScheme="auto" />
        <link rel="shortcut icon" href="/favicon.svg" />
        <meta
          name="viewport"
          content="minimum-scale=1, initial-scale=1, width=device-width, user-scalable=no"
        />
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','GTM-MFCRTXZ4');`,
          }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(websiteJsonLd) }}
        />
      </head>
      <body>
        <noscript>
          <iframe
            src="https://www.googletagmanager.com/ns.html?id=GTM-MFCRTXZ4"
            height="0"
            width="0"
            style={{ display: 'none', visibility: 'hidden' }}
          />
        </noscript>
        <MantineProvider theme={theme} defaultColorScheme="auto">
          {children}
        </MantineProvider>
      </body>
    </html>
  );
}
