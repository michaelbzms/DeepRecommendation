import {Component, ElementRef, ViewChild} from '@angular/core';
import {BackendService} from "./services/backend.service";
import {Movie} from "./models/movies";

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'ui';
  k = 10;
  movies$ = this.backendService.getMovies();
  movies: Movie[] = []
  movies_dict: { [id: string] : Movie; } = {}
  recommendations: Movie[] = [];
  files: any = {};

  @ViewChild('hidden_button') hidden_button: ElementRef<HTMLElement> | undefined;

  constructor(private backendService: BackendService) {
    this.movies$.subscribe(dict => {
      this.movies = [];
      for (const [imdbID, data] of Object.entries(dict)) {
        // @ts-ignore
        let m = new Movie(imdbID, data['primaryTitle'], data['startYear'], data['genres']);
        this.movies.push(m);
        this.movies_dict[imdbID] = m;
      }
      console.log(this.movies);
    })
  }

  round_decimal(num: number) {
    return Math.round((num + Number.EPSILON) * 1000) / 1000;
  }

  recommend() {
    // gather user ratings TODO: don't wait to do this now
    let user_ratings: object[] = []
    for (let movie of this.movies) {
      if (movie.rating != null) {
        user_ratings.push({
          'imdbID': movie.imdbID,
          'rating': movie.rating
        })
      }
    }
    if (user_ratings.length < 3) {
      alert("Not enough ratings. Needs at least three.")
      return;
    }
    console.log(user_ratings);

    let recommendations$ = this.backendService.getRecommendations(user_ratings, this.k)
    recommendations$.subscribe(recommendations => {
      console.log(recommendations);
      this.recommendations = [];
      for (let r of recommendations) {
        let m = this.movies_dict[r.imdbID];
        // Note: this changes score of original movies object but that should be ok
        m.score = r.score;
        // sort because & attention
        let ix = r.attention.map((x: any,i: any) => i);
        ix.sort((a: any, b: any) => r.attention[b] - r.attention[a]);
        m.because = ix.map((x: any) => r.because[x]);
        m.attention = ix.map((x: any) => r.attention[x]);
        this.recommendations.push(m);
      }
      if (this.hidden_button) {
        this.hidden_button.nativeElement.click();
      }
    })
  }

  wait(ms: number){
    const timeInitial : Date = new Date();
    let timeNow : Date = new Date();
    // @ts-ignore
    for ( ; timeNow - timeInitial < ms; ){
      timeNow = new Date();
    }
  }

  scroll(el: HTMLElement) {
    el.scrollIntoView({behavior: 'smooth'});
  }

  clear_ratings() {
    if (confirm('Are you sure?')) {
      for (let movie of this.movies) {
        movie.rating = null;
      }
    }
  }

  update_ratings(ratings: any) {
    for (let r of ratings) {
      try {
        this.movies_dict[r.imdbID].rating = r.rating;
      } catch (error) {
        console.error(error);
      }
    }
  }

  exportJson(obj: object[]) {
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'text/json' });
    const url = window.URL.createObjectURL(blob);
    window.open(url);
  }

  download_ratings() {
    let res: object[] = [];
    for (let m of this.movies) {
      if (m.rating) {
        res.push({"imdbID": m.imdbID, "rating": m.rating});
      }
    }
    console.log(res);
    this.exportJson(res);
  }

  onRatingsUpload(event: any) {
    const reader = new FileReader();
    reader.onload = (event) => {
      if (event.target) {
        try {
          let ratings = JSON.parse((event.target.result) as string);
          console.log('uploaded ratings:', ratings);
          this.update_ratings(ratings);
        } catch (error) {
          console.error(error);
        }
      } else {
        console.error("Null target")
      }
    };
    reader.readAsText(event.files[0]);
  }

  test(event: any) {
    console.log("Error trying to upload file");
    console.error(event)
  }
}
