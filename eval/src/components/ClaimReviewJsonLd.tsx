import type { VerificacionFrontmatter, VerdictKey } from '../types/verificacion';

const VERDICT_TO_RATING: Record<VerdictKey, { ratingValue: number; alternateName: string }> = {
  SIN_ALUCINACIONES: { ratingValue: 5, alternateName: 'Verdadero' },
  SI_PERO: { ratingValue: 4, alternateName: 'Mayormente verdadero' },
  HUMO: { ratingValue: 2, alternateName: 'Engañoso' },
  CON_ALUCINACIONES: { ratingValue: 1, alternateName: 'Falso' },
  PIDE_CONTEXTO: { ratingValue: 3, alternateName: 'Necesita contexto' },
  RUMOR_DE_X: { ratingValue: 0, alternateName: 'No comprobable' },
};

interface ClaimReviewJsonLdProps {
  frontmatter: VerificacionFrontmatter;
  slug: string;
}

export function ClaimReviewJsonLd({ frontmatter, slug }: ClaimReviewJsonLdProps) {
  const url = `https://eval.codigollm.es/verificaciones/${slug}`;
  const rating = VERDICT_TO_RATING[frontmatter.veredicto_agregado];

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'ClaimReview',
    url,
    claimReviewed: frontmatter.titulo,
    author: {
      '@type': 'Person',
      name: 'Albert Gil López',
      url: 'https://albertgilopez.com',
    },
    reviewRating: {
      '@type': 'Rating',
      ratingValue: rating.ratingValue,
      bestRating: 5,
      worstRating: 0,
      alternateName: rating.alternateName,
    },
    itemReviewed: {
      '@type': 'CreativeWork',
      author: {
        '@type': 'Person',
        name: frontmatter.autor_pieza,
      },
      datePublished: frontmatter.fecha_pieza,
    },
    datePublished: frontmatter.fecha_verificacion,
  };

  const articleJsonLd = {
    '@context': 'https://schema.org',
    '@type': 'Article',
    headline: frontmatter.titulo,
    datePublished: frontmatter.fecha_verificacion,
    dateModified: frontmatter.fecha_verificacion,
    author: {
      '@type': 'Person',
      name: 'Albert Gil López',
      url: 'https://albertgilopez.com',
      sameAs: [
        'https://linkedin.com/in/albertgilopez',
        'https://github.com/albertgilopez',
      ],
    },
    publisher: {
      '@type': 'Organization',
      name: 'Codigo LLM',
      url: 'https://www.codigollm.es',
    },
    mainEntityOfPage: url,
    inLanguage: 'es',
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(articleJsonLd) }}
      />
    </>
  );
}
