
export interface MovieInterface {
  imdbID: string;
  title: string;
  year: number;
  // TODO: more?
}

export class Movie implements MovieInterface {
  imdbID: string;
  title: string;
  year: number;

  constructor(imdbID: string, title: string, year: number) {
    this.imdbID = imdbID;
    this.title = title;
    this.year = year;
  }
}
