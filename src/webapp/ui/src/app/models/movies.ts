
export interface MovieInterface {
  imdbID: string;
  title: string;
  year: number;
  rating: number | null;
  // TODO: more?
}

export class Movie implements MovieInterface {
  imdbID: string;
  title: string;
  year: number;
  rating: number | null;

  constructor(imdbID: string, title: string, year: number) {
    this.imdbID = imdbID;
    this.title = title;
    this.year = year;
    this.rating = null;
  }
}
