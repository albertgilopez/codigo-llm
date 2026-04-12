export type VerdictKey =
  | 'SIN_ALUCINACIONES'
  | 'SI_PERO'
  | 'HUMO'
  | 'CON_ALUCINACIONES'
  | 'PIDE_CONTEXTO'
  | 'RUMOR_DE_X';

export type SubTagKey =
  | 'BENCHMARK_DEPENDIENTE'
  | 'SKYNET_FANFIC'
  | 'BIEN_ATRIBUIDO'
  | 'RUMOR_DISFRAZADO';

export interface VerdictInfo {
  key: VerdictKey;
  label: string;
  emoji: string;
  description: string;
  color: string;
}

export interface VerificacionFrontmatter {
  slug: string;
  titulo: string;
  fecha_verificacion: string;
  fecha_pieza: string;
  autor_pieza: string;
  plataforma: string;
  url_original?: string;
  veredicto_agregado: VerdictKey;
  veredicto_emoji: string;
  veredictos_contados: Record<VerdictKey, number>;
  eavi: string;
  tiempo_invertido: string;
}

export interface Verificacion {
  frontmatter: VerificacionFrontmatter;
  content: string;
  slug: string;
}
