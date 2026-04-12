import type { MetadataRoute } from 'next';
import { getAllVerificaciones } from '../src/lib/verificaciones';

const BASE_URL = 'https://eval.codigollm.es';

export default function sitemap(): MetadataRoute.Sitemap {
  const verificaciones = getAllVerificaciones();

  const verificacionEntries: MetadataRoute.Sitemap = verificaciones.map((v) => ({
    url: `${BASE_URL}/verificaciones/${v.slug}`,
    lastModified: new Date(v.frontmatter.fecha_verificacion),
    changeFrequency: 'monthly',
    priority: 0.8,
  }));

  return [
    {
      url: BASE_URL,
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 1,
    },
    {
      url: `${BASE_URL}/verificaciones`,
      lastModified: new Date(),
      changeFrequency: 'weekly',
      priority: 0.9,
    },
    {
      url: `${BASE_URL}/metodologia`,
      lastModified: new Date('2026-04-11'),
      changeFrequency: 'monthly',
      priority: 0.7,
    },
    {
      url: `${BASE_URL}/sobre`,
      lastModified: new Date('2026-04-11'),
      changeFrequency: 'monthly',
      priority: 0.5,
    },
    ...verificacionEntries,
  ];
}
