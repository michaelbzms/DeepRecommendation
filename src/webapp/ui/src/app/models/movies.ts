
export interface MovieInterface {
  imdbID: string;
  title: string;
  year: number;
  genres: string;
  rating: number | null;
  // TODO: more?
}

export class Movie implements MovieInterface {
  imdbID: string;
  title: string;
  year: number;
  genres: string;
  rating: number | null;
  score?: number;
  because?: [string];
  attention?: [number];

  constructor(imdbID: string, title: string, year: number, genres: string) {
    this.imdbID = imdbID;
    this.title = title;
    this.year = year;
    this.genres = genres.split(',').join(', ');
    this.rating = null;
  }
}
