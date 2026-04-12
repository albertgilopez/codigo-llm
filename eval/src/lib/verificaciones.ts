import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import type { Verificacion, VerificacionFrontmatter } from '../types/verificacion';

const CONTENT_DIR = path.join(process.cwd(), 'content', 'verificaciones');

export function getAllVerificaciones(): Verificacion[] {
  if (!fs.existsSync(CONTENT_DIR)) return [];

  const files = fs.readdirSync(CONTENT_DIR).filter((f) => f.endsWith('.md'));

  return files
    .map((filename) => {
      const filePath = path.join(CONTENT_DIR, filename);
      const raw = fs.readFileSync(filePath, 'utf-8');
      const { data, content } = matter(raw);
      const slug = filename.replace(/\.md$/, '');
      return {
        frontmatter: data as VerificacionFrontmatter,
        content,
        slug,
      };
    })
    .sort(
      (a, b) =>
        new Date(b.frontmatter.fecha_verificacion).getTime() -
        new Date(a.frontmatter.fecha_verificacion).getTime()
    );
}

export function getVerificacionBySlug(slug: string): Verificacion | null {
  const filePath = path.join(CONTENT_DIR, `${slug}.md`);
  if (!fs.existsSync(filePath)) return null;

  const raw = fs.readFileSync(filePath, 'utf-8');
  const { data, content } = matter(raw);
  return {
    frontmatter: data as VerificacionFrontmatter,
    content,
    slug,
  };
}

export function getAllSlugs(): string[] {
  if (!fs.existsSync(CONTENT_DIR)) return [];
  return fs
    .readdirSync(CONTENT_DIR)
    .filter((f) => f.endsWith('.md'))
    .map((f) => f.replace(/\.md$/, ''));
}
