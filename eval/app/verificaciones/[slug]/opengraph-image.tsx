import { ImageResponse } from 'next/og';
import { getVerificacionBySlug, getAllSlugs } from '../../../src/lib/verificaciones';
import { VERDICTS } from '../../../src/lib/verdicts';

export const size = { width: 1200, height: 630 };
export const contentType = 'image/png';

export function generateStaticParams() {
  return getAllSlugs().map((slug) => ({ slug }));
}

const VERDICT_COLORS: Record<string, string> = {
  SIN_ALUCINACIONES: '#2b8a3e',
  SI_PERO: '#e67700',
  HUMO: '#d9480f',
  CON_ALUCINACIONES: '#c92a2a',
  PIDE_CONTEXTO: '#1864ab',
  RUMOR_DE_X: '#862e9c',
};

export default async function OgImage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const verificacion = getVerificacionBySlug(slug);

  if (!verificacion) {
    return new ImageResponse(
      <div style={{ display: 'flex', width: '100%', height: '100%', background: '#1a1b1e', color: '#fff', alignItems: 'center', justifyContent: 'center', fontSize: 48 }}>
        eval · Codigo LLM
      </div>,
      { ...size }
    );
  }

  const { frontmatter: fm } = verificacion;
  const verdict = VERDICTS[fm.veredicto_agregado];
  const bgColor = VERDICT_COLORS[fm.veredicto_agregado] || '#1a1b1e';

  return new ImageResponse(
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        height: '100%',
        background: '#1a1b1e',
        padding: '60px',
        justifyContent: 'space-between',
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <div
          style={{
            display: 'flex',
            background: bgColor,
            color: '#fff',
            padding: '12px 24px',
            borderRadius: '8px',
            fontSize: 32,
            fontWeight: 700,
            alignSelf: 'flex-start',
          }}
        >
          {verdict?.emoji} {verdict?.label}
        </div>
        <div
          style={{
            display: 'flex',
            color: '#fff',
            fontSize: 42,
            fontWeight: 700,
            lineHeight: 1.3,
            maxWidth: '1000px',
          }}
        >
          {fm.titulo}
        </div>
      </div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-end',
          width: '100%',
        }}
      >
        <div style={{ display: 'flex', color: '#909296', fontSize: 24 }}>
          {fm.plataforma} · {fm.autor_pieza} · {fm.fecha_pieza}
        </div>
        <div style={{ display: 'flex', color: '#5c5f66', fontSize: 22 }}>
          eval · Codigo LLM
        </div>
      </div>
    </div>,
    { ...size }
  );
}
