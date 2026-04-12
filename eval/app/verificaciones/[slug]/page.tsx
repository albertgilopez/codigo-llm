import { notFound } from 'next/navigation';
import { Container, Title, Text, Stack, Group, Divider, Anchor } from '@mantine/core';
import { getVerificacionBySlug, getAllSlugs } from '../../../src/lib/verificaciones';
import { VerdictBadge } from '../../../src/components/VerdictBadge';
import { MarkdownRenderer } from '../../../src/components/MarkdownRenderer';
import { ClaimReviewJsonLd } from '../../../src/components/ClaimReviewJsonLd';
import { Breadcrumbs } from '../../../src/components/Breadcrumbs';

const BASE_URL = 'https://eval.codigollm.es';

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return getAllSlugs().map((slug) => ({ slug }));
}

export async function generateMetadata({ params }: PageProps) {
  const { slug } = await params;
  const verificacion = getVerificacionBySlug(slug);
  if (!verificacion) return {};

  const { frontmatter: fm } = verificacion;
  const title = `${fm.veredicto_emoji} ${fm.titulo}`;
  const description = `Verificación: ${fm.veredicto_emoji} ${fm.titulo}. ${Object.values(fm.veredictos_contados).reduce((a, b) => a + b, 0)} claims evaluados contra fuentes primarias.`;

  return {
    title,
    description: description.slice(0, 160),
    alternates: {
      canonical: `${BASE_URL}/verificaciones/${slug}`,
      languages: { 'es-ES': `${BASE_URL}/verificaciones/${slug}` },
    },
    openGraph: {
      title,
      description: description.slice(0, 160),
      type: 'article',
      publishedTime: fm.fecha_verificacion,
      authors: ['Albert Gil López'],
      tags: ['fact-checking', 'IA', 'verificación', fm.eavi, fm.plataforma],
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description: description.slice(0, 160),
    },
  };
}

export default async function VerificacionPage({ params }: PageProps) {
  const { slug } = await params;
  const verificacion = getVerificacionBySlug(slug);
  if (!verificacion) notFound();

  const { frontmatter, content } = verificacion;

  const breadcrumbJsonLd = {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: [
      { '@type': 'ListItem', position: 1, name: 'Inicio', item: BASE_URL },
      { '@type': 'ListItem', position: 2, name: 'Verificaciones', item: `${BASE_URL}/verificaciones` },
      { '@type': 'ListItem', position: 3, name: frontmatter.titulo },
    ],
  };

  return (
    <Container size="md" py="xl">
      <ClaimReviewJsonLd frontmatter={frontmatter} slug={slug} />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(breadcrumbJsonLd) }}
      />

      <Stack gap="lg">
        <Breadcrumbs
          items={[
            { label: 'Inicio', href: '/' },
            { label: 'Verificaciones', href: '/verificaciones' },
            { label: frontmatter.titulo },
          ]}
        />

        <Stack gap="xs">
          <Group gap="sm">
            <VerdictBadge verdict={frontmatter.veredicto_agregado} size="xl" />
            <Text size="sm" c="dimmed">
              {frontmatter.plataforma} · {frontmatter.autor_pieza}
            </Text>
          </Group>
          <Title order={1}>{frontmatter.titulo}</Title>
          {frontmatter.url_original && (
            <Anchor
              href={frontmatter.url_original}
              target="_blank"
              rel="noreferrer"
              size="sm"
            >
              Ver pieza original →
            </Anchor>
          )}
          <Group gap="xs">
            <Text size="sm" c="dimmed">
              Pieza: {frontmatter.fecha_pieza}
            </Text>
            <Text size="sm" c="dimmed">·</Text>
            <Text size="sm" c="dimmed">
              Verificación: {frontmatter.fecha_verificacion}
            </Text>
            <Text size="sm" c="dimmed">·</Text>
            <Text size="sm" c="dimmed">
              {frontmatter.tiempo_invertido}
            </Text>
          </Group>
        </Stack>

        <Divider />

        <MarkdownRenderer content={content} />
      </Stack>
    </Container>
  );
}
