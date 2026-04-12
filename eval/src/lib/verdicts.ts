import type { VerdictKey, VerdictInfo, SubTagKey } from '../types/verificacion';

export const VERDICTS: Record<VerdictKey, VerdictInfo> = {
  SIN_ALUCINACIONES: {
    key: 'SIN_ALUCINACIONES',
    label: 'SIN ALUCINACIONES',
    emoji: '✅',
    description: 'El modelo-post no alucinó.',
    color: 'green',
  },
  SI_PERO: {
    key: 'SI_PERO',
    label: 'SÍ, PERO...',
    emoji: '⭐',
    description: 'Hay letra pequeña.',
    color: 'yellow',
  },
  HUMO: {
    key: 'HUMO',
    label: 'HUMO',
    emoji: '💨',
    description: 'Cierto como slogan, engañoso como hecho.',
    color: 'orange',
  },
  CON_ALUCINACIONES: {
    key: 'CON_ALUCINACIONES',
    label: 'CON ALUCINACIONES',
    emoji: '🤖',
    description: 'Se lo alucinó un LLM.',
    color: 'red',
  },
  PIDE_CONTEXTO: {
    key: 'PIDE_CONTEXTO',
    label: 'PIDE CONTEXTO',
    emoji: '📖',
    description: 'El LLM necesita contexto.',
    color: 'blue',
  },
  RUMOR_DE_X: {
    key: 'RUMOR_DE_X',
    label: 'RUMOR DE X',
    emoji: '🐦',
    description: 'Solo existe en Twitter, Polymarket o un subreddit.',
    color: 'grape',
  },
};

export const SUB_TAGS: Record<SubTagKey, { label: string; emoji: string }> = {
  BENCHMARK_DEPENDIENTE: { label: 'BENCHMARK-DEPENDIENTE', emoji: '📊' },
  SKYNET_FANFIC: { label: 'SKYNET FANFIC', emoji: '👽' },
  BIEN_ATRIBUIDO: { label: 'BIEN ATRIBUIDO', emoji: '📎' },
  RUMOR_DISFRAZADO: { label: 'RUMOR DISFRAZADO DE HECHO', emoji: '🎭' },
};

export function getVerdict(key: VerdictKey): VerdictInfo {
  return VERDICTS[key];
}
